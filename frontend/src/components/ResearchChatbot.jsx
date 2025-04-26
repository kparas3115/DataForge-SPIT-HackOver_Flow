import { useState, useEffect, useRef } from 'react';
import { Send, Upload, MessageSquare } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function ResearchChatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [paperUploaded, setPaperUploaded] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !file.name.endsWith('.pdf')) {
      alert('Please upload a PDF file');
      return;
    }

    setFileName(file.name);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/upload-paper', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (response.ok) {
        setPaperUploaded(true);
        setMessages([
          { 
            role: 'system', 
            content: `Research paper "${file.name}" uploaded successfully. You can now ask questions about this paper!` 
          }
        ]);
      } else {
        setMessages([
          { 
            role: 'system', 
            content: `Error uploading paper: ${data.error}` 
          }
        ]);
      }
    } catch (error) {
      setMessages([
        { 
          role: 'system', 
          content: `Error uploading paper: ${error.message}` 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle sending message to chatbot
  const handleSendMessage = async () => {
    if (!input.trim() || !paperUploaded) return;
    
    const userMessage = input.trim();
    setInput('');
    
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });
      
      const data = await response.json();
      
      // Add assistant response to chat
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'system', 
        content: `Error getting response: ${error.message}` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to render message content with Markdown support
  const renderMessageContent = (content) => {
    return (
      <div className="markdown-wrapper">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-md">
      {/* Header */}
      <div className="bg-gray-50 p-4 border-b">
        <h2 className="text-xl font-semibold text-gray-800">Research Paper Chatbot</h2>
        <p className="text-sm text-gray-600">Upload a paper and ask questions about it</p>
      </div>

      {/* Chat area */}
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <MessageSquare size={48} className="mb-2" />
            <p>Upload a research paper to start chatting</p>
          </div>
        ) : (
          <>
            {messages.map((msg, index) => (
              <div 
                key={index} 
                className={`mb-4 ${
                  msg.role === 'user' 
                    ? 'flex justify-end' 
                    : 'flex justify-start'
                }`}
              >
                <div 
                  className={`p-3 rounded-lg max-w-3/4 ${
                    msg.role === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : msg.role === 'system'
                        ? 'bg-gray-200 text-gray-800'
                        : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {renderMessageContent(msg.content)}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-gray-100 p-3 rounded-lg">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Footer with input */}
      <div className="p-4 border-t">
        {!paperUploaded ? (
          <div className="flex items-center">
            <input
              type="file"
              accept=".pdf"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current.click()}
              disabled={isLoading}
              className="flex items-center justify-center w-full p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:bg-blue-300"
            >
              <Upload size={18} className="mr-2" />
              {fileName ? fileName : 'Upload Research Paper (PDF)'}
            </button>
          </div>
        ) : (
          <div className="flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about the paper..."
              className="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim()}
              className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-r-lg disabled:bg-blue-300"
            >
              <Send size={18} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}