"""
Creative AI API Endpoints
-------------------------
Exposes Claude-powered ad copy generation to the frontend.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.core.logging_config import get_logger
from app.services.ad_copy_generator import generate_ad_headlines, improve_product_description

logger = get_logger(__name__)
router = APIRouter(prefix="/creative", tags=["Creative AI"])


class HeadlineRequest(BaseModel):
    """Request for headline generation."""
    product_name: str
    keywords: List[str]
    unique_selling_points: List[str]
    target_audience: str = "general"
    tone: str = "persuasive"


class DescriptionRequest(BaseModel):
    """Request for description enhancement."""
    current_description: str
    focus_keywords: List[str]


class Asset(BaseModel):
    id: str
    url: str
    type: str  # "image", "video", "copy"
    tags: List[str]
    created_at: str
    name: str

class AssetUploadRequest(BaseModel):
    name: str
    type: str
    tags: List[str] = []



@router.post("/headlines")
async def create_headlines(request: HeadlineRequest):
    """
    Generate 5 high-converting ad headlines using Claude 3.5 Sonnet.
    
    Example request:
    ```json
    {
        "product_name": "Wireless Bluetooth Headphones",
        "keywords": ["wireless headphones", "bluetooth earbuds", "noise cancelling"],
        "unique_selling_points": ["40-hour battery", "Premium sound quality", "Comfortable fit"],
        "target_audience": "music lovers",
        "tone": "professional"
    }
    ```
    """
    try:
        result = await generate_ad_headlines(
            product_name=request.product_name,
            keywords=request.keywords,
            unique_selling_points=request.unique_selling_points,
            target_audience=request.target_audience,
            tone=request.tone
        )
        return result
    except Exception as e:
        logger.error(f"Headline generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/description")
async def enhance_description(request: DescriptionRequest):
    """
    Rewrite a product description to be more engaging and SEO-optimized.
    Uses Claude 3.5 Sonnet for natural, persuasive copy.
    """
    try:
        result = await improve_product_description(
            current_description=request.current_description,
            focus_keywords=request.focus_keywords
        )
        return result
    except Exception as e:
        logger.error(f"Description enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets", response_model=List[Asset])
async def list_assets():
    """
    List available creative assets.
    """
    # Mock assets
    return [
        Asset(
            id="asset_001",
            url="https://lh3.googleusercontent.com/aida-public/AB6AXuD5gnd4r4PiHB3YF59m5wO_z2XEzWI1IGvkfceYeKUMp_v2lNo4ymj0LmHGu8s40_0SelJdxnmWjMR1b2RzVVMMeDbvH03RZIDZau-4DdsDTfUorU8G0k32-xR2TZIWRioB_GHVN5CTfuKawuOyYOTfLUWAXx1-s2ARMHTTxXr4ksPyBQhN71kJ1P6JvDwgFwRPfpJs4aQykuXomxxM1jHg1q16PuJy9icSulGcJ3j1BN7mYuSmBmWOyksrSqlCo11wESF2iqXD8eFM",
            type="image",
            tags=["lifestyle", "neon", "cyberpunk"],
            created_at="2023-10-27T10:00:00Z",
            name="IMG_GEN_04.jpg"
        ),
        Asset(
            id="asset_002",
            url="https://lh3.googleusercontent.com/aida-public/AB6AXuAl0r9lCNFTgybx0bvcQFbvYwEp83wMQEy6-TddjRPZLBcw__onm8oK36thfZgGbTx7I9Bg55PFm4IZ9Dg2c8g61jGdmyW-G_Tzc3PLw96QDmIfNYd41IukGNH4KE5XDNG8rv3EXdKjUWbh0Fm0h6cStmrdVQYng0pDlz8t9he0ER0lG-s7M4EuQoNqsliDnIvGQh5XShsbAZ2qLv9Yyg9J9UB0M6Lc7K1Stk_Q-AIUM5eHmwvfaGeiulPvfd66HXhprSdkr41hizmY",
            type="image",
            tags=["product", "minimalist", "white"],
            created_at="2023-10-26T14:30:00Z",
            name="PROD_WHITE_02.png"
        ),
        Asset(
            id="asset_003",
            url="https://lh3.googleusercontent.com/aida-public/AB6AXuDtsvWmi7PFSkCvVz4zBBzuhMkkivMLB2eOpQmYU1Xy61aI-avy8uOtU0YKf7xzMH5M57XDE2_AB_6towOldknAz6JgPNPMZQFnfnwYVco9AvlcmAwKzYbcX_0Zo8_EVMSs9B6-NCzRDPPsfchLx0aX4iiOgxDSGcq5FWYrYQ7WhtCx9iy-0BhCz6EymOY9t4oboguT_uixLwXGqx_hEYwEtdTcWmNJ5ueds-lrhVH1F_9AVQ-OlhfCktbjqyHqIFpYBNHWXx4d0ZuY",
            type="image",
            tags=["lifestyle", "desk", "wood"],
            created_at="2023-10-25T09:15:00Z",
            name="LIFESTYLE_DESK_01.jpg"
        )
    ]

@router.post("/upload", response_model=Asset)
async def upload_asset(request: AssetUploadRequest):
    """
    Mock endpoint for uploading assets.
    """
    return Asset(
        id="asset_new_001",
        url="https://via.placeholder.com/300",
        type=request.type,
        tags=request.tags,
        created_at="2023-10-28T12:00:00Z",
        name=request.name
    )
