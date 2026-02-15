
from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

router = APIRouter()

# --- Models ---

class MetricTrend(BaseModel):
    value: float
    trend_percent: float
    trend_direction: str # up, down, flat

class PerformanceMetric(BaseModel):
    label: str
    value: str
    trend: MetricTrend
    sub_label: Optional[str] = None
    color: str # primary, accent-cyan, accent-pink, yellow-400

class ChartDataPoint(BaseModel):
    label: str
    sales: float
    acos: float

class ProductPerformance(BaseModel):
    name: str
    asin: str
    image_url: str
    sales: float
    orders: int
    acos: float
    trend_7d: str # simple svg path or direction for frontend to render

class DashboardData(BaseModel):
    client_name: str
    last_updated: str
    kpis: List[PerformanceMetric]
    chart_data: List[ChartDataPoint]
    budget_pacing: dict
    top_products: List[ProductPerformance]
    ai_logs: List[dict]

# --- Mock Data ---

MOCK_DASHBOARD = DashboardData(
    client_name="Aura Cosmetics",
    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    kpis=[
        PerformanceMetric(
            label="Total Sales",
            value="$14,250",
            trend=MetricTrend(value=12, trend_percent=12, trend_direction="up"),
            color="primary"
        ),
        PerformanceMetric(
            label="ACoS",
            value="24.5%",
            trend=MetricTrend(value=-2, trend_percent=2, trend_direction="down"),
            sub_label="Target: 25.0%",
            color="accent-cyan"
        ),
        PerformanceMetric(
            label="Ad Spend",
            value="$3,400",
            trend=MetricTrend(value=5, trend_percent=5, trend_direction="up"),
            sub_label="Daily Avg: $485",
            color="accent-pink"
        ),
        PerformanceMetric(
            label="ROAS",
            value="4.2x",
            trend=MetricTrend(value=1.5, trend_percent=1.5, trend_direction="up"),
            sub_label="Profitable > 3.5x",
            color="yellow-400"
        )
    ],
    chart_data=[
        ChartDataPoint(label="Mon", sales=8400, acos=28),
        ChartDataPoint(label="Tue", sales=9200, acos=26),
        ChartDataPoint(label="Wed", sales=10500, acos=24),
        ChartDataPoint(label="Thu", sales=11200, acos=25),
        ChartDataPoint(label="Fri", sales=12400, acos=22),
        ChartDataPoint(label="Sat", sales=13800, acos=23),
        ChartDataPoint(label="Sun", sales=14250, acos=21)
    ],
    budget_pacing={
        "total": 500,
        "spent": 325,
        "remaining": 175,
        "percentage": 65,
        "status": "normal"
    },
    top_products=[
        ProductPerformance(
            name="Radiance Face Serum",
            asin="B08XQW2P1Z",
            image_url="https://lh3.googleusercontent.com/aida-public/AB6AXuAoZbWlNnufhSzzmXpxYMZKD00wAbA0Is_6bCk7ywaQN9jtD7ySlDf7SOlX7QPZovSZu604r4j0sOUHTBd8PAYtfivHvn_P1KOQdupFInrTxhG2I5g6ajtYMmr8i55gcQPU1mjWQYmyTW96PV898H73cj4RHCWA98zWemOJGkayv8gKvbnVYHVqq4vAzD4HyGwQ4PSAp-FJ5iIUtoUVkm3Fk5yrWtipdLySkVQY_ADmnEV8hoKXeQsvBQ6VPWnBhOUQkFqtTC4Sm53W",
            sales=4250,
            orders=142,
            acos=21.5,
            trend_7d="up"
        ),
        ProductPerformance(
            name="Night Repair Cream",
            asin="B07YTR5M9K",
            image_url="https://lh3.googleusercontent.com/aida-public/AB6AXuAFuruU1DvUDeNCLSOxTAFKqZPtg9PJcJ5ir6XAxQEeaB_p6fQMrBiLYZkNvOUTrs8_fUyBriLvBpShILqlRfcs8OK817kYvd5-UGIJYCuqAeeQzTE7GE4faW0ludsupBSqmblBLOwxPZuA4eR3m_31zdXgUEXXaKR0Wu57mjqnmi4uK2bOc87kb56r_yLWgfnQ7JKSTMBwf2TEkv9DotPUkXRp7_ub-4pOyq0H3C0Hg-R2vN5U7u0gdO7mM_GEYOWA0OenRJ-jvdnc",
            sales=3105,
            orders=98,
            acos=28.2,
            trend_7d="stable"
        ),
        ProductPerformance(
            name="Velvet Matte Lipstick",
            asin="B09ZK4L2N1",
            image_url="https://lh3.googleusercontent.com/aida-public/AB6AXuA4jT3oPIh50Zi1LSPOgSTEXy_wT151INQPbjzN_VK_KGRd4ma-xWA1FeryFHXQtXBu7-eamC6dhAx41GVzU2mFGm48OhsRLUDY3DGY23r-gsAyVjzcpmA18a8uzMJhBNdi7gtHvc2V4W6KLbuz5PSZ5zup4lCUHM6NW8VxFVSPfAgtmItcE6smGQUps8udRIxQ7-co_K2g_LzPSg0xJXnNVBbk3WK4KZPKGbZ4Dib0vRTe1XDNXTjFyvtWDty3aoUoWAzO8AxLwWJm",
            sales=1890,
            orders=112,
            acos=34.1,
            trend_7d="down"
        )
    ],
    ai_logs=[
        {"time": "10:42:15", "type": "BID_OPT_SUCCESS", "message": 'Keyword "organic serum" bid increased to $1.45 (+5%) to maintain ToS.', "color": "green"},
        {"time": "10:15:33", "type": "NEG_KW_ADDED", "message": 'Added "cheap makeup" to negative phrase match (Campaign: Auto_Catchall).', "color": "yellow"},
        {"time": "09:55:01", "type": "BUDGET_CHECK", "message": "Routine budget verification complete. All campaigns within variance.", "color": "blue"},
        {"time": "09:30:12", "type": "BID_OPT_SUCCESS", "message": 'Decreased bid on "matte lipstick" by -10% due to high ACoS.', "color": "green"}
    ]
)

# --- Endpoints ---

@router.get("/dashboard/{client_id}", response_model=DashboardData)
async def get_client_dashboard(client_id: str):
    # In a real app, we'd fetch data for the specific client_id
    data = MOCK_DASHBOARD
    data.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return data
