import os
import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class RecommendationItem(BaseModel):
    title: str = Field(description="Short title (max 5 words)")
    action: str = Field(description="Concrete action (under 15 words)")
    rationale: str = Field(description="Business or risk rationale (under 15 words)")
    priority: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Priority level")

class ExplainabilityReport(BaseModel):
    prediction_confidence_pct: float = Field(description="Confidence percentage (0-100)")
    executive_summary: str = Field(description="Executive overview (max 2 sentences, under 35 words).")
    key_risk_drivers: List[str] = Field(description="Max 2 short risk bullets (under 12 words each).")
    positive_factors: List[str] = Field(description="Max 2 short positive bullets (under 12 words each).")

class LLMCopilotOutput(BaseModel):
    explainability: ExplainabilityReport
    recommendations: List[RecommendationItem]

class FleetSenseEngine:
    def __init__(self, data_dir: str = './'):
        self.data_dir = data_dir

    def calculate_risk_analysis(self, equipment: str, quantity: int, site_id: str, booking_date: str) -> Dict[str, Any]:
        predicted_demand = 52
        available_units = 45
        gap = max(0, quantity - available_units)
        capacity_ratio = round(available_units / max(1, quantity), 2)
        confidence = 100.0 if gap == 0 else max(50.0, 100.0 - (gap * 10))
        risk_level = "LOW" if capacity_ratio >= 1.5 else ("MEDIUM" if capacity_ratio >= 1.0 else "HIGH")

        return {
            "inquiry": {
                "equipment_type": equipment,
                "quantity_requested": quantity,
                "site_id": site_id,
                "booking_date": booking_date,
                "site_city": "Bengaluru"
            },
            "metrics": {
                "predicted_demand": predicted_demand,
                "expected_available": available_units,
                "shortage_gap": gap,
                "capacity_ratio": capacity_ratio,
                "fulfillment_confidence_pct": confidence
            },
            "risk_analysis": {
                "overall_risk_level": risk_level,
                "risk_breakdown_pct": {
                    "turnaround_maintenance": 56.0,
                    "weather_rain": 24.0,
                    "return_delays": 20.0
                }
            },
            "underlying_context_factors": {
                "rain_probability": "42%",
                "weather_condition": "Moderate Rain",
                "units_in_turnaround_maintenance": 48,
                "average_on_time_return_probability": "92%",
                "active_rentals_at_risk_of_delay": 2
            }
        }

    def generate_rule_recommendations(self, risk_payload: Dict[str, Any]) -> List[Dict[str, str]]:
        recs = []
        metrics = risk_payload["metrics"]
        if metrics["shortage_gap"] > 0:
            recs.append({
                "priority": "HIGH",
                "action": "Accelerate turnaround maintenance on 2 units by 24 hours.",
                "reason": f"Shortage gap of {metrics['shortage_gap']} unit(s) identified."
            })
            recs.append({
                "priority": "MEDIUM",
                "action": "Suggest shifting rental start date by +2 days.",
                "reason": "Aligns delivery with incoming equipment return windows."
            })
        else:
            recs.append({
                "priority": "LOW",
                "action": f"Approve reservation for {risk_payload['inquiry']['quantity_requested']} {risk_payload['inquiry']['equipment_type']}s immediately.",
                "reason": f"Fulfillment confidence is {metrics['fulfillment_confidence_pct']}% with zero shortage gap."
            })
            recs.append({
                "priority": "LOW",
                "action": "Proceed with standard servicing on idle turnaround units.",
                "reason": f"Abundant surplus capacity ({metrics['capacity_ratio']}x) supports standard buffer schedules."
            })
        return recs

    def generate_copilot_insights(self, equipment: str, quantity: int, site_id: str, booking_date: str) -> LLMCopilotOutput:
        risk_payload = self.calculate_risk_analysis(equipment, quantity, site_id, booking_date)
        risk_payload["rule_recommendations"] = self.generate_rule_recommendations(risk_payload)

        payload_compact_json = json.dumps(risk_payload, separators=(',', ':'))
        SYSTEM_PROMPT = """You are Lead Operations AI Copilot for 'FleetSense AI'.
TASK: Analyze operational payload and return a strictly structured assessment.
CONSTRAINTS: Rely strictly on provided JSON numbers. Never fabricate stats or dates."""

        USER_PROMPT = f"[CONTEXT]: FleetSense AI Inquiry\n[PAYLOAD]: {payload_compact_json}\n\nGenerate structured Copilot report."

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-flash-latest',
                    contents=USER_PROMPT,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=LLMCopilotOutput,
                        temperature=0.2,
                    ),
                )
                if getattr(response, 'parsed', None) is not None:
                    return response.parsed
                elif response.text:
                    return LLMCopilotOutput.model_validate_json(response.text)
            except Exception as e:
                print(f"⚠️ Gemini API Call Error: {e}")

        return LLMCopilotOutput(
            explainability=ExplainabilityReport(
                prediction_confidence_pct=risk_payload["metrics"]["fulfillment_confidence_pct"],
                executive_summary=f"Inquiry for {quantity} {equipment}s at {site_id} carries {risk_payload['risk_analysis']['overall_risk_level']} risk with {risk_payload['metrics']['capacity_ratio']}x coverage.",
                key_risk_drivers=["Rain probability is 42%.", "48 units in turnaround."],
                positive_factors=[f"Capacity ratio of {risk_payload['metrics']['capacity_ratio']}x.", "High historical on-time return rate."]
            ),
            recommendations=[
                RecommendationItem(title="Directives", action=r["action"], rationale=r["reason"], priority=r["priority"])
                for r in risk_payload["rule_recommendations"]
            ]
        )
