// Type definitions for result-related interfaces

// Basic result information
export interface ResultInfo {
    id: string;
    media_type: 'image' | 'audio' | 'video';
    is_fake: boolean;
    confidence_score: number;
    created_at: string;
  }
  
  // Detailed result with all analysis data
  export interface DetailedResult extends ResultInfo {
    detection_details: Record<string, any>;
    models_used: Record<string, string>;
    visualizations?: VisualizationData;
  }
  
// Visualization data for different analysis types
export interface VisualizationData {
  heatmap?: HeatmapData;
  temporal?: TemporalData;
  frequency?: Record<string, any>;
  spectral_discontinuity?: SpectralData;
  voice_consistency?: VoiceConsistencyData;
  silence_analysis?: SilenceData;
  confidence_gauge?: ConfidenceGaugeData;
}

// Heatmap visualization data
export interface HeatmapData {
  url: string;
  width: number;
  height: number;
  regions: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    label?: string;
  }>;
}

// Temporal analysis visualization data
export interface TemporalData {
  url?: string;
  timestamps: number[];
  values: number[];
  threshold: number;
}

// Spectral discontinuity visualization for audio
export interface SpectralData {
  url: string;
  splice_times: number[];
  threshold: number;
}

// Voice consistency visualization for audio
export interface VoiceConsistencyData {
  url: string;
  segment_diffs: number[];
  mean_diff: number;
}

// Silence segments visualization for audio
export interface SilenceData {
  url: string;
  segments: Array<{
    start: number;
    end: number;
    duration: number;
  }>;
  total_duration: number;
}

// Confidence gauge visualization
export interface ConfidenceGaugeData {
  url: string;
  score: number;
}
  
  // Result statistics
  export interface ResultStatistics {
    period_days: number;
    total_results: number;
    media_type_distribution: {
      image: number;
      audio: number;
      video: number;
    };
    real_count: number;
    fake_count: number;
    fake_percentage: number;
    average_confidence: number;
  }
  
  // Query options for results
  export interface ResultQuery {
    task_id?: string;
    result_id?: string;
  }
  
  // Status response for a query
  export interface ResultStatus {
    status: string;
    progress: number;
    message?: string;
    result_id?: string;
  }
