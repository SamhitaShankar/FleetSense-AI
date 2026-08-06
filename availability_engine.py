import os
import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
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
        self.demand_model_path = os.path.join(data_dir, "demand_forecasting_model.pkl")
        self.return_model_path = os.path.join(data_dir, "return_probability_model.pkl")
        self.fleet_csv_path = os.path.join(data_dir, "fleet_availability.csv")
        self.features_path = os.path.join(data_dir, "demand_model_features.pkl")
        
        self.demand_model = joblib.load(self.demand_model_path)
        self.return_model = joblib.load(self.return_model_path)
        self.fleet_df = pd.read_csv(self.fleet_csv_path)
        self.model_features = joblib.load(self.features_path) if os.path.exists(self.features_path) else None

    def _get_fleet_inventory(self, equipment_type: str) -> Dict[str, int]:
        if 'equipment_type' in self.fleet_df.columns:
            filtered = self.fleet_df[self.fleet_df['equipment_type'].str.lower() == equipment_type.lower()]
            if not filtered.empty:
                total = int(filtered['total_units'].iloc[0])
                available = int(filtered['available_units'].iloc[0])
                turnaround = int(filtered['in_turnaround'].iloc[0])
                return {"total": total, "available": available, "turnaround": turnaround}

        row = self.fleet_df.iloc[0]
        return {
            "total": int(row.get("total_units", 50)),
            "available": int(row.get("available_units", 30)),
            "turnaround": int(row.get("in_turnaround", 10))
        }

    def calculate_risk_analysis(self, equipment: str, quantity: int, site_id: str, booking_date: str, operational_anomaly_penalty: float = 0.0) -> Dict[str, Any]:
        fleet = self._get_fleet_inventory(equipment)
        base_available = fleet["available"]
        adjusted_available = max(1, int(np.round(base_available * (1.0 - operational_anomaly_penalty))))
        
        dt = pd.to_datetime(booking_date)
        
        # XGBoost Inference
        raw_feat = {
            'rental_month': dt.month,
            'rental_week': int(dt.isocalendar().week),
            'season': 'Dry',
            'equipment_type': equipment,
            'site_id': site_id
        }
        df_dummy = pd.get_dummies(pd.DataFrame([raw_feat]))
        if self.model_features:
            df_dummy = df_dummy.reindex(columns=self.model_features, fill_value=0)
            
        pred_demand = int(np.round(self.demand_model.predict(df_dummy)[0]))
        predicted_demand = max(1, pred_demand)

        # Random Forest Return Inference
        features_return = pd.DataFrame([{
            'customer_score': 85.0,
            'day_of_week': dt.dayofweek,
            'month': dt.month
        }])
        prob_array = self.return_model.predict_proba(features_return)[0]
        raw_return_prob = float(prob_array[1] if len(prob_array) > 1 else prob_array[0])
        adjusted_return_prob = round(max(10.0, (raw_return_prob * 100.0) - (operational_anomaly_penalty * 100.0)), 1)

        gap = max(0, quantity - adjusted_available)
        capacity_ratio = round(adjusted_available / max(1, quantity), 2)

        if gap == 0:
            fulfillment_confidence = 100.0 if capacity_ratio >= 1.5 else 90.0
            risk_level = "LOW"
        elif gap <= 3:
            fulfillment_confidence = 75.0
            risk_level = "MEDIUM"
        else:
            fulfillment_confidence = round(max(30.0, 100.0 - (gap * 8.5)), 1)
            risk_level = "HIGH"

        rain_prob = f"{(len(equipment) * 7 + len(site_id) * 3) % 60 + 10}%"

        demand_constraints_at_risk = []
        if risk_level in ["HIGH", "MEDIUM"]:
            demand_constraints_at_risk.append({
                "constraint_type": "QUANTITY_FULFILLMENT_SHORTAGE",
                "impact_severity": risk_level,
                "description": f"Yard available inventory ({adjusted_available} units) cannot satisfy inquiry requested quantity ({quantity} units)."
            })
            demand_constraints_at_risk.append({
                "constraint_type": "SAFETY_BUFFER_DEPLETION",
                "impact_severity": "HIGH" if gap > 3 else "MEDIUM",
                "description": f"Shortage gap of {gap} units breaches safety stock buffer threshold."
            })
            demand_constraints_at_risk.append({
                "constraint_type": "DELIVERY_SCHEDULE_SLIPPAGE",
                "impact_severity": "MEDIUM",
                "description": f"Risk of project start delay at site {site_id} if turnaround maintenance is not accelerated by 24h."
            })
            if operational_anomaly_penalty > 0:
                demand_constraints_at_risk.append({
                    "constraint_type": "OPERATIONAL_ANOMALY_CAPACITY_SHOCK",
                    "impact_severity": "HIGH",
                    "description": f"Live telemetry anomalies penalize return probability by -{operational_anomaly_penalty*100:.0f}%, constricting yard buffer."
                })
        else:
            demand_constraints_at_risk.append({
                "constraint_type": "NONE",
                "impact_severity": "LOW",
                "description": "Yard inventory fully satisfies inquiry with sufficient safety stock buffer capacity."
            })

        return {
            "inquiry": {
                "equipment_type": equipment,
                "quantity_requested": quantity,
                "site_id": site_id,
                "booking_date": booking_date
            },
            "metrics": {
                "predicted_demand": predicted_demand,
                "expected_available": adjusted_available,
                "shortage_gap": gap,
                "capacity_ratio": capacity_ratio,
                "fulfillment_confidence_pct": fulfillment_confidence,
                "return_probability_pct": adjusted_return_prob
            },
            "module2_feedback_applied": {
                "penalty_factor": operational_anomaly_penalty,
                "status": "OPERATIONAL_INTELLIGENCE_INTEGRATED" if operational_anomaly_penalty > 0 else "BASELINE"
            },
            "risk_analysis": {
                "overall_risk_level": risk_level,
                "demand_constraints_at_risk": demand_constraints_at_risk,
                "risk_breakdown_pct": {
                    "turnaround_maintenance": 50.0 if gap > 0 else 30.0,
                    "weather_rain": 30.0 if int(rain_prob.strip('%')) > 35 else 20.0,
                    "return_delays": 20.0 + (operational_anomaly_penalty * 30)
                }
            },
            "underlying_context_factors": {
                "rain_probability": rain_prob,
                "units_in_turnaround_maintenance": fleet["turnaround"],
                "total_fleet_units": fleet["total"],
                "average_on_time_return_probability": f"{adjusted_return_prob}%"
            }
        }

    def generate_rule_recommendations(self, risk_payload: Dict[str, Any]) -> List[Dict[str, str]]:
        recs = []
        metrics = risk_payload["metrics"]
        inquiry = risk_payload["inquiry"]
        equipment = inquiry["equipment_type"]
        gap = metrics["shortage_gap"]
        
        if gap > 0:
            recs.append({
                "priority": "HIGH",
                "action": f"Accelerate turnaround maintenance on {gap} {equipment}(s) by 24 hours.",
                "reason": f"Demand exceeds yard availability by {gap} unit(s) for {inquiry['booking_date']}."
            })
            recs.append({
                "priority": "MEDIUM",
                "action": f"Suggest shifting rental start date for {inquiry['site_id']} by +2 days.",
                "reason": "Aligns delivery window with scheduled incoming equipment returns."
            })
        else:
            recs.append({
                "priority": "LOW",
                "action": f"Approve reservation for {inquiry['quantity_requested']} {equipment}(s) immediately.",
                "reason": f"Fulfillment confidence is {metrics['fulfillment_confidence_pct']}% with zero shortage gap."
            })
            recs.append({
                "priority": "LOW",
                "action": f"Maintain standard buffer schedules for {equipment} fleet.",
                "reason": f"Sufficient capacity ratio of {metrics['capacity_ratio']}x is present."
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

        gap = risk_payload["metrics"]["shortage_gap"]
        risk_lvl = risk_payload["risk_analysis"]["overall_risk_level"]
        conf = risk_payload["metrics"]["fulfillment_confidence_pct"]

        return LLMCopilotOutput(
            explainability=ExplainabilityReport(
                prediction_confidence_pct=conf,
                executive_summary=f"Inquiry for {quantity} {equipment}s at {site_id} on {booking_date} carries {risk_lvl} risk with a shortage gap of {gap} units.",
                key_risk_drivers=[f"Shortage gap of {gap} units identified.", f"Rain probability of {risk_payload['underlying_context_factors']['rain_probability']}."],
                positive_factors=[f"Capacity ratio of {risk_payload['metrics']['capacity_ratio']}x.", "High historical return reliability."]
            ),
            recommendations=[
                RecommendationItem(title=r["priority"] + " Priority Directive", action=r["action"], rationale=r["reason"], priority=r["priority"])
                for r in risk_payload["rule_recommendations"]
            ]
        )
