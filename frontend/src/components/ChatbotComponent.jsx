// src/components/Chatbot/ChatbotComponent.jsx
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const ChatbotComponent = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! Upload a research paper, and I can answer questions about it.' }
  ]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [currentPaper, setCurrentPaper] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload-paper', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const paperName = file.name;
      setCurrentPaper(paperName);
      setMessages(prev => [
        ...prev,
        { role: 'user', content: `Uploaded: ${paperName}` },
        { role: 'assistant', content: `I've processed "${paperName}". What would you like to know about this paper?` }
      ]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'There was an error uploading your file. Please try again.' }
      ]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isTyping) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsTyping(true);

    try {
      const response = await axios.post('/api/chat', {
        message: userMessage,
      });

      setMessages(prev => [...prev, { role: 'assistant', content: response.data.response }]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error processing your question. Please try again.' }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-md">
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-semibold text-gray-700">Research Assistant</h3>
        {currentPaper && <span className="text-xs text-gray-500">Current paper: {currentPaper}</span>}
      </div>
      
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.map((message, index) => (
          <div key={index} className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block p-3 rounded-lg ${
              message.role === 'user' 
                ? 'bg-blue-500 text-white rounded-br-none' 
                : 'bg-gray-100 text-gray-800 rounded-bl-none'
            }`}>
              {message.content}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="text-left mb-4">
            <div className="inline-block p-3 rounded-lg bg-gray-100 text-gray-800 rounded-bl-none">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-4 border-t">
        <div className="flex mb-2">
          <button 
            onClick={triggerFileUpload}
            className="flex items-center justify-center px-3 py-1 mr-2 text-xs bg-gray-200 hover:bg-gray-300 rounded-md transition"
            disabled={isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload Paper'}
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".pdf"
            className="hidden"
          />
        </div>
        <div className="flex">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about the paper..."
            className="flex-1 p-2 border rounded-l-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-300"
            rows={2}
            disabled={isTyping}
          />
          <button
            onClick={handleSendMessage}
            disabled={!input.trim() || isTyping}
            className={`px-4 rounded-r-lg ${
              !input.trim() || isTyping 
                ? 'bg-gray-300 cursor-not-allowed' 
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatbotComponent;