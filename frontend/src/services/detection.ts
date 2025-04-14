import { apiRequest, uploadFile as uploadFileApi } from './api';
import axios from 'axios';

interface DetectionParams {
    media_type: 'image' | 'audio' | 'video';
    detailed_analysis?: boolean;
    confidence_threshold?: number;
}

interface DetectionResponse {
    task_id: string;
    status: string;
    media_type: string;
    estimated_time: number;
}

interface StatusResponse {
    status: string;
    progress: number;
    message?: string;
    result_id?: string;
}

interface UploadResponse {
    filename: string;
    size: number;
    content_type: string;
    message: string;
}

// Upload file and start detection
export const uploadFile = async (
    file: File,
    params: DetectionParams
): Promise<DetectionResponse> => {
    try {
        // Create FormData for direct submission to detection endpoint
        const formData = new FormData();
        formData.append('file', file);
        
        // Serialize detection parameters
        const detectionParams = JSON.stringify({
            media_type: params.media_type,
            detailed_analysis: params.detailed_analysis ?? false,
            confidence_threshold: params.confidence_threshold ?? 0.5
        });
        formData.append('detection_params', detectionParams);
        
        console.log('Uploading file for detection:', {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            params: detectionParams
        });

        // Connect to our backend running on port 8001
        const response = await fetch('http://localhost:8001/upload', {
            method: 'POST',
            body: formData,
            // CORS should be properly configured on the backend now
            headers: {
                'Accept': 'application/json'
            }
        }).then(async res => {
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`Upload error (${res.status}): ${errorText}`);
            }
            return res.json();
        });
        
        console.log('Detection response:', response);
        return response;
    } catch (error) {
        console.error('Upload and detect error:', error);
        throw error;
    }
};

// Check status of detection task
export const checkDetectionStatus = async (taskId: string): Promise<StatusResponse> => {
    try {
        // Connect directly to our backend running on port 8001
        const response = await fetch(`http://localhost:8001/status/${taskId}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Status check failed: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Status check response:', data);
        return data;
    } catch (error) {
        console.error('Error checking detection status:', error);
        throw error;
    }
};

// Get recent detection results
export const getRecentResults = async (limit: number = 10): Promise<any[]> => {
    return apiRequest<any[]>(`/detection/results?limit=${limit}`);
};

// Get specific detection result
export const getDetectionResult = async (resultId: string): Promise<any> => {
    // For the demo-result special case
    if (resultId === 'demo-result-123') {
        // Return a simulated detection result
        return {
            id: 'demo-result-123',
            status: 'completed',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            file_name: '9e0792fa4d1bc15db2e1b1440d0ef66b.jpg',
            file_hash: '9e0792fa4d1bc15db2e1b1440d0ef66b',
            media_type: 'image',
            result: {
                prediction: 'real',
                confidence: 0.98,
                analyzed_at: new Date().toISOString(),
                processing_time: 1.2,
                model_used: 'FaceForensics',
                visual_explanation: null
            }
        };
    }
    
    // For real task IDs, use the API
    return apiRequest<any>(`/detection/results/${resultId}`);
};
