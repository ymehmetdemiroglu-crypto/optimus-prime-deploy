# Data Models Document

## 1. Conceptual Data Model
The system revolves around **Sellers** managing **Campaigns** for specific **ASINs**, guided by **AI Insights**.

## 2. TypeScript Interfaces (Frontend)

```typescript
// User / Seller Context
interface SellerProfile {
  id: string;
  name: string;
  marketplace: 'US' | 'UK' | 'DE';
  currency: string;
}

// The Core Product Entity
interface ProductASIN {
  asin: string;
  title: string;
  imageUrl: string;
  price: number;
  inventoryLevel: number;
}

// Advertising Campaign
interface Campaign {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'archived';
  aiMode: 'manual' | 'auto_pilot' | 'aggressive_growth' | 'profit_guard';
  dailyBudget: number;
  spend: number;
  sales: number;
  acos: number; // Advertising Cost of Sales %
}

// Dashboard Time-Series Data
interface PerformanceMetric {
  timestamp: string; // ISO Date
  organicSales: number;
  adSales: number;
  spend: number;
  impressions: number;
}

// Chat / AI Interaction
interface ChatMessage {
  id: string;
  sender: 'user' | 'grok';
  content: string;
  timestamp: string;
  // Optional: Structured data actions embedded in chat
  actionSuggestion?: {
    type: 'apply_bid_adjustment' | 'create_campaign';
    payload: any;
  };
}
```

## 3. Pydantic Models (Backend)

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum

class AIStrategy(str, Enum):
    MANUAL = "manual"
    AUTO_PILOT = "auto_pilot"
    AGGRESSIVE = "aggressive_growth"
    PROFIT = "profit_guard"

class CampaignModel(BaseModel):
    id: str
    name: str
    status: str
    ai_mode: AIStrategy
    acos: float
    spend: float
    sales: float

class ChatRequest(BaseModel):
    message: str
    context_asin: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    sender: str = "grok"
    content: str
    timestamp: datetime
```
