"""schemas.py - Pydantic request and response schemas for PathSnap API."""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    trip_id: str = Field(..., description="8-digit zero-padded trip identifier, e.g. '00000000'", min_length=8, max_length=8)


class PredictionMetrics(BaseModel):
    top1_accuracy: Optional[float] = Field(None, description="Observation-level Top-1 accuracy against ground truth")
    route_continuity: Optional[float] = Field(None, description="Road-network direct transition continuity ratio")
    observation_macro_f1: Optional[float] = Field(None, description="Multi-class observation macro F1 across local arcs")
    route_jaccard: Optional[float] = Field(None, description="Jaccard set overlap of reconstructed vs ground truth route")
    n_observations: int = Field(..., description="Number of GPS observations in trajectory")
    n_decoded_arcs: int = Field(..., description="Number of road segments in reconstructed route")


class TimingBreakdown(BaseModel):
    feature_generation_s: float = Field(..., description="Latency of KDTree candidate generation & feature extraction")
    model_scoring_s: float = Field(..., description="Latency of classifier scoring pipeline")
    viterbi_decoding_s: float = Field(..., description="Latency of HMM/Viterbi sequential decoding & route reconstruction")
    total_inference_s: float = Field(..., description="Total wall-clock inference time in seconds")


class PredictionResponse(BaseModel):
    trip_id: str
    status: str
    metrics: Optional[PredictionMetrics] = None
    predicted_route: List[int] = Field(..., description="Chronological sequence of reconstructed traversed arc IDs")
    route_coordinates: List[List[float]] = Field(..., description="Ordered list of [lat, lon] coordinates along predicted path")
    gps_coordinates: List[List[float]] = Field(..., description="Ordered list of [lat, lon] coordinates of raw GPS fixes")
    inference_time_seconds: float
    timing_breakdown: TimingBreakdown
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: str
    features_count: int
    timestamp: str


class TripsResponse(BaseModel):
    trips: List[str]
    partition: str
    total_trips: int
    message: str
