import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from availability_engine import FleetSenseEngine, LLMCopilotOutput

app = FastAPI(
    title="FleetSense AI API",
    description="Enterprise Decision Support Backend for Heavy Machinery Rentals",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_dir = os.environ.get("DATA_DIR", "./")
engine = FleetSenseEngine(data_dir=data_dir)

class CopilotRequest(BaseModel):
    equipment: str = "Excavator"
    quantity: int = 10
    site_id: str = "SITE003"
    booking_date: str = "2026-08-20"

@app.get("/")
def health_check():
    return {"status": "online", "system": "FleetSense AI Backend Engine", "version": "1.0.0"}

@app.get("/api/v1/dashboard")
def get_dashboard_kpis(equipment: str = Query("Excavator"), site_id: str = Query("SITE003")):
    risk_payload = engine.calculate_risk_analysis(equipment, 10, site_id, "2026-08-20")
    return {
        "predicted_demand": risk_payload["metrics"]["predicted_demand"],
        "expected_available": risk_payload["metrics"]["expected_available"],
        "shortage_gap": risk_payload["metrics"]["shortage_gap"],
        "fulfillment_confidence_pct": risk_payload["metrics"]["fulfillment_confidence_pct"],
        "overall_risk_level": risk_payload["risk_analysis"]["overall_risk_level"],
        "capacity_ratio": risk_payload["metrics"]["capacity_ratio"]
    }

@app.get("/api/v1/dashboard/charts")
def get_dashboard_charts(equipment: str = Query("Excavator")):
    return {
        "demand_trend": [
            {"date": "2026-08-16", "demand": 42, "available": 45},
            {"date": "2026-08-17", "demand": 44, "available": 45},
            {"date": "2026-08-18", "demand": 48, "available": 45},
            {"date": "2026-08-19", "demand": 50, "available": 45},
            {"date": "2026-08-20", "demand": 52, "available": 45},
            {"date": "2026-08-21", "demand": 49, "available": 45}
        ],
        "fleet_status_breakdown": {
            "active_rentals": 120,
            "available_in_yard": 45,
            "under_turnaround_maintenance": 48,
            "in_transit": 12
        },
        "risk_factor_breakdown": {
            "maintenance_turnaround": 56.0,
            "weather_rain": 24.0,
            "return_delays": 20.0
        }
    }

@app.get("/api/v1/equipment/{equipment_id}")
def get_equipment_details(equipment_id: str):
    return {
        "equipment_id": equipment_id,
        "model": "CAT 320 Heavy Excavator",
        "status": "In Yard (Available)",
        "telemetry": {
            "operating_hours": 3420,
            "fuel_level_pct": 88,
            "hydraulic_pressure_psi": 3150,
            "health_score_pct": 94
        },
        "last_maintenance_date": "2026-07-28",
        "next_scheduled_service": "2026-09-15"
    }

@app.post("/api/v1/copilot/insights", response_model=LLMCopilotOutput)
def get_ai_copilot_insights(req: CopilotRequest):
    try:
        return engine.generate_copilot_insights(
            equipment=req.equipment,
            quantity=req.quantity,
            site_id=req.site_id,
            booking_date=req.booking_date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
