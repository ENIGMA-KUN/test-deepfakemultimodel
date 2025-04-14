import { apiRequest } from './api';

interface ResultQuery {
    task_id?: string;
    result_id?: string;
}

interface ResultStatus {
    status: string;
    progress: number;
    message?: string;
    result_id?: string;
}

// Query result status by task ID or result ID
export const queryResult = async (query: ResultQuery): Promise<ResultStatus> => {
    return apiRequest<ResultStatus>('/results/query', {
        method: 'POST',
        body: JSON.stringify(query),
    });
};

// Fetch detailed result by result ID
export const fetchResultDetails = async (resultId: string): Promise<any> => {
    try {
        console.log('Fetching result details for:', resultId);
        
        // Connect directly to our backend running on port 8001
        const response = await fetch(`http://localhost:8001/results/${resultId}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to fetch result details: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Result details response:', data);
        return data;
    } catch (error) {
        console.error('Error fetching result details:', error);
        throw error;
    }
};