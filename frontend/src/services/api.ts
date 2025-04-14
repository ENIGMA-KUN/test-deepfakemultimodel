// Base API configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

console.log('Using API URL:', API_URL); // For debugging

// Generic HTTP request handler
export async function apiRequest<T = any>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const headers = {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
    };

    const config = {
        ...options,
        headers,
    };

    const url = `${API_URL}${endpoint}`;
    console.log('Requesting:', url); // For debugging
    
    try {
        const response = await fetch(url, config);

        if (!response.ok) {
            // Handle different error status codes
            if (response.status === 404) {
                throw new Error('Resource not found');
            }

            if (response.status === 401) {
                throw new Error('Unauthorized access');
            }

            if (response.status === 429) {
                throw new Error('Too many requests, please try again later');
            }

            // Try to get error details from response
            try {
                const errorData = await response.json();
                throw new Error(errorData.detail || `API error: ${response.status}`);
            } catch (e) {
                throw new Error(`API error: ${response.status}`);
            }
        }

        // Return JSON response, or empty object if no content
        if (response.status !== 204) {
            return await response.json();
        }

        return {} as T;
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Upload file with form data
export async function uploadFileRequest<T = any>(
    endpoint: string,
    file: File,
    data?: Record<string, any>,
    options: RequestInit = {}
): Promise<T> {
    const formData = new FormData();
    
    // Debugging - log what we're uploading
    console.log('Uploading file:', file.name, file.type, file.size);
    
    // Ensure the file is properly appended with the correct field name
    formData.append('file', file);
    
    // Add media_type directly for backend compatibility
    if (data && data.media_type) {
        formData.append('media_type', data.media_type);
    }
    
    // Add detailed_analysis directly for backend compatibility
    if (data && data.hasOwnProperty('detailed_analysis')) {
        formData.append('detailed_analysis', data.detailed_analysis.toString());
    }
    
    // Add confidence_threshold directly for backend compatibility
    if (data && data.hasOwnProperty('confidence_threshold')) {
        formData.append('confidence_threshold', data.confidence_threshold.toString());
    }
    
    // Also add the full detection_params as JSON for React frontend
    if (data) {
        formData.append('detection_params', JSON.stringify(data));
    }

    const config = {
        ...options,
        method: 'POST',
        body: formData,
        // Don't set Content-Type header, let the browser set it with boundary
        headers: {
            ...(options.headers || {}),
        },
    };

    return apiRequest<T>(endpoint, config);
}
