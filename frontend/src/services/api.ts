import axios, { AxiosRequestConfig } from 'axios';

// Base API configuration
const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false
});

interface ApiRequestOptions {
  method?: string;
  body?: any;
  headers?: Record<string, string>;
}

// Generic HTTP request handler
export async function apiRequest<T = any>(
    endpoint: string,
    options: ApiRequestOptions = {}
): Promise<T> {
    try {
        // Create request config
        const requestConfig: AxiosRequestConfig = {
            url: endpoint,
            method: options.method || 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...(options.headers || {}),
            }
        };

        // Add data if present
        if (options.body) {
            requestConfig.data = options.body;
        }

        console.log('Making API request:', {
            endpoint,
            method: requestConfig.method,
            data: requestConfig.data
        });
        
        const response = await api.request(requestConfig);

        console.log('API response:', response.data);
        return response.data;
    } catch (error) {
        console.error('API request failed:', {
            endpoint,
            error: error instanceof Error ? error.message : 'Unknown error',
            details: axios.isAxiosError(error) ? error.response?.data : undefined
        });
        throw error;
    }
}

// Upload file with form data
export const uploadFile = async (file: File) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    console.log('Uploading file:', {
      name: file.name,
      size: file.size,
      type: file.type
    });

    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    console.log('Upload response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Upload error:', {
      error: error instanceof Error ? error.message : 'Unknown error',
      details: axios.isAxiosError(error) ? error.response?.data : undefined
    });
    throw error;
  }
};
