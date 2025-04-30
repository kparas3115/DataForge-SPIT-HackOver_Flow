import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = ({ onUploadComplete, sessionId }) => { // Add sessionId prop
  const [uploadStatus, setUploadStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // const handleFileUpload = async (event) => {
  //   const file = event.target.files[0];
  //   if (!file) return;

  //   setIsLoading(true);
  //   setUploadStatus('Uploading to S3...');

  //   try {
  //     // First upload to S3
  //     const formData = new FormData();
  //     formData.append('file', file);
      
  //     const s3Response = await axios.post('http://localhost:3000/api/upload/single', formData, {
  //       headers: { 
  //         'Content-Type': 'multipart/form-data',
  //       }
  //     });

  //     setUploadStatus('Processing PDF...');

  //     // Then send URL to Python backend for processing
  //     const processResponse = await axios.post('http://localhost:5000/api/upload_paper', {
  //       url: s3Response.data.url,
  //       session_id: sessionId // Use the passed sessionId
  //     }, {
  //       headers: {
  //         'Content-Type': 'application/json'
  //       }
  //     });

  //     setUploadStatus('Upload complete!');
      
  //     if (onUploadComplete) {
  //       onUploadComplete({
  //         success: true,
  //         filename: s3Response.data.filename,
  //         url: s3Response.data.url,
  //         processingResult: processResponse.data
  //       });
  //     }
  //   } catch (error) {
  //     console.error('Upload error:', error);
  //     setUploadStatus(`Upload failed: ${error.response?.data?.error || error.message}`);
      
  //     if (onUploadComplete) {
  //       onUploadComplete({
  //         success: false,
  //         error: error.response?.data?.error || error.message
  //       });
  //     }
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", "your-session-id"); // or generate dynamically
  
    try {
      const response = await fetch("http://localhost:5000/api/upload_paper", {
        method: "POST",
        body: formData,
      });
  
      const result = await response.json();
      if (response.ok) {
        console.log("Upload success:", result);
      } else {
        console.error("Upload failed:", result.error);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };
  

  return (
    <div className="w-full max-w-md p-4 border rounded-lg shadow">
      <label className="block mb-4">
        <span className="text-gray-700">Upload Research Paper (PDF)</span>
        <input
          type="file"
          accept="application/pdf"
          onChange={handleFileUpload}
          disabled={isLoading}
          className="mt-1 block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100
          "
        />
      </label>
      {uploadStatus && (
        <div className={`mt-2 text-sm ${isLoading ? 'text-blue-600' : 'text-green-600'}`}>
          {uploadStatus}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
