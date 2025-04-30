import React, { useState } from 'react';

const JournalRecommender = () => {
  const [title, setTitle] = useState('');
  const [abstract, setAbstract] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [error, setError] = useState(null);
  const [reportError, setReportError] = useState(null);
  const [topK, setTopK] = useState(5);
  const [reportUrl, setReportUrl] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/recommend-journals', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, abstract, top_k: topK }),
      });

      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }

      const data = await response.json();
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    setIsGeneratingReport(true);
    setReportError(null);
    setReportUrl(null);

    try {
      const response = await fetch('http://localhost:5000/api/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, abstract, top_k: topK }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const data = await response.json();
      if (data.success && data.download_url) {
        setReportUrl(data.download_url);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      setReportError(err.message);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-2xl font-bold mb-4">ResearchSaathi Journal Recommender</h2>
      <p className="text-gray-600 mb-4">
        Enter your paper details to find suitable journals for publication or generate a comprehensive PDF report.
      </p>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-1">
            Paper Title
          </label>
          <input
            id="title"
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter the title of your paper"
            required
          />
        </div>
        
        <div className="mb-4">
          <label htmlFor="abstract" className="block text-sm font-medium text-gray-700 mb-1">
            Abstract
          </label>
          <textarea
            id="abstract"
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded"
            rows="6"
            placeholder="Paste your paper abstract here"
            required
          />
        </div>
        
        <div className="mb-4">
          <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-1">
            Number of Recommendations
          </label>
          <select
            id="topK"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded"
          >
            <option value="3">3</option>
            <option value="5">5</option>
            <option value="10">10</option>
          </select>
        </div>
        
        <div className="flex flex-wrap gap-3">
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded"
            disabled={isLoading}
          >
            {isLoading ? 'Finding Journals...' : 'Find Matching Journals'}
          </button>
          
          <button
            type="button"
            onClick={handleGenerateReport}
            className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded"
            disabled={isGeneratingReport || !title || !abstract}
          >
            {isGeneratingReport ? 'Generating Report...' : 'Generate PDF Report'}
          </button>
        </div>
      </form>
      
      {error && (
        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded">
          Error: {error}
        </div>
      )}
      
      {reportError && (
        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded">
          Report Error: {reportError}
        </div>
      )}
      
      {reportUrl && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded">
          <h3 className="text-lg font-medium text-green-800">Report Generated Successfully!</h3>
          <p className="mt-2 text-green-700">
            Your comprehensive journal recommendation report is ready.
          </p>
          <a 
            href={`http://localhost:5000${reportUrl}`} 
            className="mt-3 inline-block bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded"
            target="_blank" 
            rel="noopener noreferrer"
          >
            Download PDF Report
          </a>
        </div>
      )}
      
      {recommendations.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-3">Recommended Journals</h3>
          <div className="space-y-4">
            {recommendations.map((journal, index) => (
              <div key={index} className="border border-gray-200 rounded p-4 hover:bg-gray-50">
                <div className="flex justify-between items-start">
                  <h4 className="text-lg font-medium">{index + 1}. {journal.name}</h4>
                  <span className="bg-blue-100 text-blue-800 text-sm px-2 py-1 rounded">
                    Score: {journal.score.toFixed(4)}
                  </span>
                </div>
                <p className="text-gray-600 text-sm mt-1">Publisher: {journal.publisher}</p>
                {journal.impact_factor && (
                  <p className="text-gray-600 text-sm mt-1">
                    Impact Factor: {typeof journal.impact_factor === 'number' ? journal.impact_factor.toFixed(2) : journal.impact_factor}
                  </p>
                )}
                {journal.h_index && (
                  <p className="text-gray-600 text-sm mt-1">
                    H-Index: {journal.h_index}
                  </p>
                )}
                {journal.url && (
                  <a 
                    href={journal.url} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="text-blue-600 hover:underline text-sm mt-1 block"
                  >
                    Visit Journal Website
                  </a>
                )}
                {journal.subjects && journal.subjects.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs text-gray-500">Subjects:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {journal.subjects.slice(0, 5).map((subject, idx) => (
                        <span key={idx} className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded">
                          {subject}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default JournalRecommender;