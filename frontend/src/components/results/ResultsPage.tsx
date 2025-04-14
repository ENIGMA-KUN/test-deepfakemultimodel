import React, { useState } from 'react';
import ConfidenceScore from './ConfidenceScore';
import HeatmapView from './HeatmapView';
import TemporalAnalysis from './TemporalAnalysis';
import FrequencyAnalysis from './FrequencyAnalysis';
import ExplanationPanel from './ExplanationPanel';
import Button from '../common/Button';

interface ResultsPageProps {
    result: any; // Replace with proper type
    mediaType: 'image' | 'audio' | 'video';
    onNewAnalysis: () => void;
    className?: string;
}

const ResultsPage: React.FC<ResultsPageProps> = ({
    result,
    mediaType,
    onNewAnalysis,
    className = '',
}) => {
    const [activeTab, setActiveTab] = useState<'summary' | 'visual' | 'technical'>('summary');

    return (
        <div className={`bg-gray-900 rounded-lg border border-gray-700 shadow-xl overflow-hidden ${className}`}>
            {/* Header */}
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4 border-b border-gray-700">
                <div className="flex flex-wrap items-center justify-between">
                    <h2 className="text-xl font-bold text-white flex items-center">
                        <span className={`h-3 w-3 rounded-full mr-2 ${result.result?.prediction === 'fake' ? 'bg-red-500' : 'bg-green-500'}`}></span>
                        {result.result?.prediction === 'fake' ? 'Manipulated Content Detected' : 'Content Appears Authentic'}
                    </h2>

                    <div className="flex space-x-2 mt-2 sm:mt-0">
                        <Button
                            variant="secondary"
                            size="sm"
                            onClick={onNewAnalysis}
                        >
                            New Analysis
                        </Button>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="bg-gray-800 border-b border-gray-700">
                <nav className="flex overflow-x-auto">
                    <button
                        onClick={() => setActiveTab('summary')}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-200 focus:outline-none
              ${activeTab === 'summary'
                                ? 'text-blue-500 border-b-2 border-blue-500'
                                : 'text-gray-400 hover:text-gray-200'}
            `}
                    >
                        Summary
                    </button>

                    <button
                        onClick={() => setActiveTab('visual')}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-200 focus:outline-none
              ${activeTab === 'visual'
                                ? 'text-blue-500 border-b-2 border-blue-500'
                                : 'text-gray-400 hover:text-gray-200'}
            `}
                    >
                        Visual Analysis
                    </button>

                    <button
                        onClick={() => setActiveTab('technical')}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-200 focus:outline-none
              ${activeTab === 'technical'
                                ? 'text-blue-500 border-b-2 border-blue-500'
                                : 'text-gray-400 hover:text-gray-200'}
            `}
                    >
                        Technical Details
                    </button>
                </nav>
            </div>

            {/* Content */}
            <div className="p-4">
                {activeTab === 'summary' && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        <div className="lg:col-span-1">
                            <ConfidenceScore
                                score={result.result?.confidence || 0}
                                isFake={result.result?.prediction === 'fake'}
                                mediaType={mediaType}
                            />
                        </div>

                        <div className="lg:col-span-2">
                            <ExplanationPanel
                                result={result}
                                mediaType={mediaType}
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'visual' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {(mediaType === 'image' || mediaType === 'video') && result.result?.visual_explanation && (
                            <div className="md:col-span-2">
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <h3 className="text-lg font-medium text-white mb-3">Visual Analysis</h3>
                                    <div className="flex justify-center">
                                        <img 
                                            src={`http://localhost:8001${result.result.visual_explanation}`} 
                                            alt="Visual Analysis" 
                                            className="max-w-full rounded-lg shadow-lg"
                                        />
                                    </div>
                                    {result.result.prediction === 'fake' && (
                                        <div className="mt-4 p-3 bg-gray-700 bg-opacity-40 rounded-lg text-gray-300">
                                            <h4 className="text-white font-medium mb-2">Key Areas of Inconsistency</h4>
                                            <p>The highlighted regions indicate potential manipulation. 
                                            {mediaType === 'image' ? 'Inconsistencies in texture and lighting patterns are typical indicators of deepfakes.' : 
                                             mediaType === 'video' ? 'Temporal inconsistencies and unnatural movements are typical indicators of video deepfakes.' : 
                                             'Spectral anomalies and unnatural voice patterns are typical indicators of voice synthesis.'}
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                        
                        {(!result.result?.visual_explanation && result.result?.prediction === 'fake') && (
                            <div className="md:col-span-2 p-8 text-center">
                                <div className="text-gray-400">
                                    <p>No visual analysis available for this media type.</p>
                                </div>
                            </div>
                        )}
                        
                        {result.result?.prediction === 'real' && (
                            <div className="md:col-span-2 p-8 text-center">
                                <div className="text-gray-400">
                                    <p>No visual analysis needed - content appears to be authentic.</p>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'technical' && (
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <h3 className="text-lg font-medium text-white mb-3">Technical Details</h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Models Used */}
                            <div className="bg-gray-900 p-4 rounded-lg">
                                <h4 className="text-gray-300 font-medium mb-2">Model Information</h4>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Model:</span>
                                        <span className="text-gray-300">{result.result?.model_used || 'Unknown'}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Detection Method:</span>
                                        <span className="text-gray-300">{result.result?.technical_details?.detection_method || 'Unknown'}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Processing Time:</span>
                                        <span className="text-gray-300">{result.result?.processing_time || 0} seconds</span>
                                    </div>
                                </div>
                            </div>

                            {/* Detection Details */}
                            <div className="bg-gray-900 p-4 rounded-lg">
                                <h4 className="text-gray-300 font-medium mb-2">Detection Details</h4>
                                <div className="space-y-2">
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-400">Confidence:</span>
                                        <div className="flex items-center">
                                            <div className="w-32 h-2 bg-gray-700 rounded-full mr-2">
                                                <div
                                                    className={`h-2 rounded-full ${result.result?.prediction === 'fake' ? 'bg-red-500' : 'bg-green-500'}`}
                                                    style={{ width: `${(result.result?.confidence || 0) * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="text-gray-300">{Math.round((result.result?.confidence || 0) * 100)}%</span>
                                        </div>
                                    </div>
                                    
                                    {/* Media-specific metrics */}
                                    {mediaType === 'video' && (
                                        <>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Frames:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.frame_count || 0}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">FPS:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.fps || 0}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Duration:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.duration || 0}s</span>
                                            </div>
                                        </>
                                    )}
                                    
                                    {mediaType === 'audio' && (
                                        <>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Sample Rate:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.sample_rate || 'Unknown'}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Duration:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.duration || 0}s</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Spectral Analysis:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.spectral_coherence || 'Unknown'}</span>
                                            </div>
                                        </>
                                    )}
                                    
                                    {mediaType === 'image' && (
                                        <>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Resolution:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.resolution || 'Unknown'}</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Color Analysis:</span>
                                                <span className="text-gray-300">{result.result?.technical_details?.color_analysis || 'Unknown'}</span>
                                            </div>
                                        </>
                                    )}
                                </div>
                            </div>

                            {/* Manipulation Indicators */}
                            {result.result?.key_indicators && result.result.key_indicators.length > 0 && (
                                <div className="bg-gray-900 p-4 rounded-lg md:col-span-2">
                                    <h4 className="text-gray-300 font-medium mb-2">Detected Manipulation Indicators</h4>
                                    <ul className="list-disc pl-5 text-gray-400 space-y-1">
                                        {result.result.key_indicators.map((indicator: string, index: number) => (
                                            <li key={index}>{indicator}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            
                            {/* Frame Analysis for Videos */}
                            {result.result?.frame_analysis && result.result.frame_analysis.length > 0 && (
                                <div className="bg-gray-900 p-4 rounded-lg md:col-span-2">
                                    <h4 className="text-gray-300 font-medium mb-2">Frame Analysis</h4>
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-700">
                                            <thead>
                                                <tr>
                                                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Frame</th>
                                                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Confidence</th>
                                                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Key Points</th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-gray-900 divide-y divide-gray-800">
                                                {result.result.frame_analysis.map((frame: any, index: number) => (
                                                    <tr key={index}>
                                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{frame.frame}</td>
                                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{frame.confidence}%</td>
                                                        <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-300">{frame.key_points}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}
                            
                            {/* Raw JSON for debugging */}
                            <div className="bg-gray-900 p-4 rounded-lg md:col-span-2">
                                <h4 className="text-gray-300 font-medium mb-2">Raw Response Data</h4>
                                <pre className="text-gray-400 text-sm overflow-auto p-2 bg-black bg-opacity-30 rounded max-h-60">
                                    {JSON.stringify(result, null, 2)}
                                </pre>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ResultsPage;