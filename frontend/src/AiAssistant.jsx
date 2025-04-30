import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { 
  Bot, 
  X, 
  Send, 
  Loader2, 
  MessageSquare, 
  FileText,
  Trash2,
  Download,
  Maximize2, 
  Minimize2,
  Upload,
  Info,
  Clock,
  Search
} from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import FileUpload from './FileUpload';
import axios from 'axios';

const AiAssistant = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => localStorage.getItem('aiSessionId') || uuidv4());
  const messagesEndRef = useRef(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [pastConversations, setPastConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(sessionId);
  const inputRef = useRef(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [recentUploads, setRecentUploads] = useState([]);

  // Initialize session
  useEffect(() => {
    localStorage.setItem('aiSessionId', sessionId);
    loadChatHistory();
  }, [sessionId]);

  // Load past conversations
  useEffect(() => {
    const loadPastConversations = () => {
      const conversations = Object.keys(localStorage)
        .filter(key => key.startsWith('chat_'))
        .map(key => ({
          id: key.replace('chat_', ''),
          messages: JSON.parse(localStorage.getItem(key) || '[]'),
          lastMessage: new Date(JSON.parse(localStorage.getItem(key) || '[]')[0]?.timestamp || Date.now())
        }))
        .sort((a, b) => b.lastMessage - a.lastMessage);
      setPastConversations(conversations);

      // Set mock recent uploads for demo
      setRecentUploads([
        { id: 'rec1', name: 'Research on Climate Change.pdf', date: '2025-02-28' },
        { id: 'rec2', name: 'Machine Learning Survey.pdf', date: '2025-02-25' },
        { id: 'rec3', name: 'Neural Networks Paper.pdf', date: '2025-02-20' },
      ]);
    };
    loadPastConversations();
  }, [messages]);

  // Auto scroll to bottom of chat when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const loadChatHistory = async () => {
    try {
      // Load messages from localStorage
      const savedMessages = localStorage.getItem(`chat_${sessionId}`);
      if (savedMessages) {
        setMessages(JSON.parse(savedMessages));
      }

      // Load papers from server
      const response = await axios.get(`http://localhost:5000/api/list_papers`, {
        params: { session_id: sessionId }
      });
      
      if (response.data.papers) {
        setFiles(response.data.papers.map(paper => ({
          id: paper.paper_id,
          name: paper.filename,
          uploadTime: paper.upload_time
        })));
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
      addMessage({
        type: 'error',
        content: 'Error loading chat history'
      });
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: uuidv4(),
      type: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setInput(''); // Clear input immediately for better UX
    
    setMessages(prev => {
      const updated = [...prev, userMessage];
      localStorage.setItem(`chat_${sessionId}`, JSON.stringify(updated));
      return updated;
    });
    
    setIsLoading(true);

    try {
      // Send message to AI endpoint
      const response = await axios.post('http://localhost:5000/api/chat', {
        message: input,
        session_id: sessionId
      });

      // Add AI response to chat
      const aiMessage = {
        id: uuidv4(),
        type: 'assistant',
        content: response.data.response,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => {
        const updated = [...prev, aiMessage];
        localStorage.setItem(`chat_${sessionId}`, JSON.stringify(updated));
        return updated;
      });

    } catch (error) {
      console.error('Chat error:', error);
      addMessage({
        type: 'error',
        content: `Error: ${error.response?.data?.error || error.message}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (message) => {
    setMessages(prev => {
      const updated = [...prev, { ...message, id: uuidv4(), timestamp: new Date().toISOString() }];
      localStorage.setItem(`chat_${sessionId}`, JSON.stringify(updated));
      return updated;
    });
  };

  const handleFileUpload = async (result) => {
    if (result.success) {
      const newFile = {
        id: uuidv4(),
        name: result.filename,
        uploadTime: new Date().toISOString(),
        url: result.url
      };
      
      setFiles(prev => {
        const updated = [...prev, newFile];
        localStorage.setItem(`files_${sessionId}`, JSON.stringify(updated));
        return updated;
      });

      // Reload papers list after upload
      loadChatHistory();

      addMessage({
        type: 'system',
        content: `Successfully uploaded and processed ${result.filename}`,
        files: [newFile]
      });
    }
  };

  const clearSession = async () => {
    try {
      await axios.post('http://localhost:5000/api/clear_session', {
        session_id: sessionId
      });
      
      localStorage.removeItem(`chat_${sessionId}`);
      localStorage.removeItem(`files_${sessionId}`);
      setMessages([]);
      setFiles([]);
      setInput('');
      const newSessionId = uuidv4();
      localStorage.setItem('aiSessionId', newSessionId);
    } catch (error) {
      console.error('Error clearing session:', error);
      addMessage({
        type: 'error',
        content: 'Error clearing session'
      });
    }
  };

  const downloadChatHistory = () => {
    const history = messages.map(m => {
      return `[${new Date(m.timestamp).toLocaleString()}] ${m.type}: ${m.content}`;
    }).join('\n');
    
    const blob = new Blob([history], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-history-${sessionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const deleteFile = async (fileId) => {
    try {
      // You would typically call an API endpoint here to delete the file from the server
      // await axios.delete(`http://localhost:5000/api/delete_paper/${fileId}`);
      
      setFiles(files.filter(f => f.id !== fileId));
      localStorage.setItem(
        `files_${sessionId}`,
        JSON.stringify(files.filter(f => f.id !== fileId))
      );
      
      addMessage({
        type: 'system',
        content: `File deleted successfully`
      });
    } catch (error) {
      console.error('Error deleting file:', error);
      addMessage({
        type: 'error',
        content: 'Error deleting file'
      });
    }
  };

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => {
        inputRef.current.focus();
      }, 300);
    }
  }, [isOpen]);

  // Filter files based on search term
  const filteredFiles = files.filter(file => 
    file.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="fixed bottom-8 right-8 z-50">
      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className={`fixed ${
              isExpanded 
                ? 'inset-0' // Full screen mode
                : 'bottom-4 right-4 w-[95vw] sm:w-[85vw] md:w-[600px] lg:w-[800px] h-[85vh]' // Normal mode
            } bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl border border-gray-200/50 flex flex-col overflow-hidden`}
          >
            {/* Header */}
            <div className="p-4 border-b flex justify-between items-center bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-t-2xl">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-white/20 rounded-xl">
                  <Bot className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold tracking-tight">AI Research Assistant</h3>
                  <p className="text-xs text-blue-100 font-medium">Always here to help</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="hover:bg-white/20 text-white h-8 w-8"
                >
                  {isExpanded ? 
                    <Minimize2 className="h-4 w-4" /> : 
                    <Maximize2 className="h-4 w-4" />
                  }
                </Button>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={() => setIsOpen(false)}
                  className="hover:bg-white/20 text-white h-8 w-8"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="flex-1 overflow-hidden">
              <Tabs defaultValue="chat" className="h-full flex flex-col">
                <TabsList className="flex-shrink-0 px-3 py-2 border-b bg-white grid grid-cols-2">
                  <TabsTrigger value="chat" className="flex gap-1 items-center text-xs data-[state=active]:bg-blue-50 data-[state=active]:text-blue-700">
                    <MessageSquare className="h-3 w-3" />
                    Chat
                  </TabsTrigger>
                  <TabsTrigger value="files" className="flex gap-1 items-center text-xs data-[state=active]:bg-blue-50 data-[state=active]:text-blue-700">
                    <FileText className="h-3 w-3" />
                    Files ({files.length})
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="chat" className="flex-1 flex flex-col overflow-visible">
                  <div className="flex-1 overflow-y-auto">
                    <div className="p-4 space-y-6 max-w-4xl mx-auto pb-4"> 
                      {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center p-6 my-6">
                          <div className="bg-blue-100/50 rounded-full p-6 mb-4">
                            <Bot className="h-16 w-16 text-blue-500" />
                          </div>
                          <h3 className="text-xl font-bold text-gray-800 mb-2">Welcome to AI Research Assistant</h3>
                          <p className="text-gray-600 max-w-lg mb-6 text-base">I can help you analyze research papers, answer questions, and more. Try uploading a document or asking me a question!</p>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full max-w-2xl">
                            {[
                              "What's the main finding in my paper?",
                              "Can you summarize this research?",
                              "What are the limitations mentioned?"
                            ].map((suggestion) => (
                              <button 
                                key={suggestion}
                                onClick={() => setInput(suggestion)}
                                className="bg-white hover:bg-blue-50 transition-colors p-3 rounded-xl shadow-sm border border-blue-100 text-left text-gray-700 hover:text-blue-700 flex items-start text-sm"
                              >
                                <MessageSquare className="h-4 w-4 mr-2 text-blue-500 flex-shrink-0 mt-0.5" />
                                <span>{suggestion}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <>
                          {messages.map((message, index) => (
                            <motion.div
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ duration: 0.3 }}
                              key={message.id}
                              className={`flex gap-4 ${message.type === 'user' ? 'justify-end' : ''}`}
                            >
                              {message.type !== 'user' && (
                                <Avatar className="h-10 w-10 border-2 border-white shadow-md flex-shrink-0">
                                  <AvatarFallback className="bg-gradient-to-br from-blue-500 to-blue-600 text-white">AI</AvatarFallback>
                                  <AvatarImage src="/ai-avatar.png" alt="AI" />
                                </Avatar>
                              )}
                              <div
                                className={`rounded-2xl p-4 max-w-[85%] md:max-w-[75%] shadow-sm ${
                                  message.type === 'user'
                                    ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white'
                                    : message.type === 'error'
                                    ? 'bg-red-50 text-red-600 border border-red-100'
                                    : message.type === 'system'
                                    ? 'bg-gray-100 text-gray-600'
                                    : 'bg-white border border-gray-100'
                                }`}
                              >
                                <div className="prose prose-sm">
                                  {message.content}
                                </div>
                                {message.files && (
                                  <div className="mt-3 flex flex-wrap gap-2">
                                    {message.files.map(file => (
                                      <Badge key={file.id} variant="secondary" className={`${
                                        message.type === 'user' ? 'bg-white/20 text-white' : 'bg-blue-50 text-blue-800'
                                      } px-3 py-1 text-xs font-medium rounded-full`}>
                                        <FileText className="h-3.5 w-3.5 mr-1.5 inline" />
                                        {file.name}
                                      </Badge>
                                    ))}
                                  </div>
                                )}
                                <div className={`text-xs mt-2 ${message.type === 'user' ? 'text-blue-100' : 'text-gray-400'}`}>
                                  {index > 0 && message.type !== messages[index-1].type && 
                                    new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
                                  }
                                </div>
                              </div>
                              {message.type === 'user' && (
                                <Avatar className="h-10 w-10 border-2 border-white shadow-md flex-shrink-0">
                                  <AvatarFallback className="bg-gradient-to-br from-purple-500 to-purple-600 text-white">ME</AvatarFallback>
                                  <AvatarImage src="/user-avatar.png" alt="User" />
                                </Avatar>
                              )}
                            </motion.div>
                          ))}
                          <div ref={messagesEndRef} />
                        </>
                      )}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="files" className="flex-1 flex flex-col">
                  <div className="grid grid-cols-12 h-full">
                    {/* Left Panel (File Browser) - Expanded to use more space */}
                    <div className="col-span-7 bg-white flex flex-col h-full border-r">
                      <div className="p-3 border-b flex items-center justify-between">
                        <h3 className="text-base font-semibold text-gray-700 flex items-center gap-2">
                          <FileText className="h-4 w-4" />
                          Document Library
                        </h3>
                        <div className="relative w-48">
                          <Input 
                            placeholder="Search files..." 
                            className="pl-8 h-8 text-sm"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                          />
                          <Search className="h-4 w-4 absolute left-2 top-2 text-gray-400" />
                        </div>
                      </div>
                      
                      {/* Tabs for file organization */}
                      <div className="border-b">
                        <div className="flex">
                          <button className="py-2 px-4 text-sm font-medium border-b-2 border-blue-500 text-blue-700">
                            All Files ({files.length})
                          </button>
                          <button className="py-2 px-4 text-sm font-medium text-gray-500 hover:text-gray-700">
                            Recent
                          </button>
                          <button className="py-2 px-4 text-sm font-medium text-gray-500 hover:text-gray-700">
                            Favorites
                          </button>
                        </div>
                      </div>
                      
                      <div className="flex-1 overflow-y-auto p-3">
                        {filteredFiles.length > 0 ? (
                          <div className="grid grid-cols-2 gap-3">
                            {filteredFiles.map((file) => (
                              <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                key={file.id}
                                className="p-3 rounded-lg border hover:border-blue-200 bg-white hover:bg-blue-50 transition-colors shadow-sm"
                              >
                                <div className="flex items-center mb-2">
                                  <div className="w-10 h-10 rounded bg-blue-100 flex items-center justify-center mr-3">
                                    <FileText className="h-5 w-5 text-blue-500" />
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <p className="font-medium text-sm truncate">{file.name}</p>
                                    <p className="text-xs text-gray-500 flex items-center gap-1">
                                      <Clock className="h-3 w-3 inline" />
                                      {new Date(file.uploadTime).toLocaleDateString()}
                                    </p>
                                  </div>
                                </div>
                                <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-100">
                                  {file.url && (
                                    <a 
                                      href={file.url} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                      className="text-xs text-blue-500 hover:text-blue-600 flex items-center"
                                    >
                                      <Info className="h-3 w-3 mr-1" />
                                      View
                                    </a>
                                  )}
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => deleteFile(file.id)}
                                    className="h-6 p-1 text-xs hover:bg-red-50 text-red-500 rounded hover:text-red-600"
                                  >
                                    <Trash2 className="h-3 w-3 mr-1" />
                                    Delete
                                  </Button>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-center py-8 bg-gray-50 rounded-lg border border-dashed border-gray-200">
                            <FileText className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                            <h3 className="text-base font-medium text-gray-600 mb-1">No files found</h3>
                            <p className="text-sm text-gray-500">
                              {searchTerm ? "Try a different search term" : "Upload files to get started"}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Right Panel (Upload & Info) */}
                    <div className="col-span-5 flex flex-col h-full">
                      <div className="h-full overflow-y-auto bg-blue-50 p-4 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-transparent hover:scrollbar-thumb-gray-400">
                        <div className="space-y-4 pb-4">
                          <div className="bg-white p-4 rounded-lg border shadow-sm">
                            <h3 className="font-semibold text-gray-700 mb-3 flex items-center">
                              <Upload className="h-4 w-4 mr-2 text-blue-500" />
                              Upload Research Paper
                            </h3>
                            <FileUpload 
                              sessionId={sessionId}
                              onUploadComplete={handleFileUpload}
                            />
                            
                            <div className="mt-4 pt-4 border-t border-gray-100">
                              <p className="text-xs text-gray-500 mb-2">Supported formats: PDF, DOCX, TXT</p>
                              <p className="text-xs text-gray-500">Maximum file size: 10MB</p>
                            </div>
                          </div>
                          
                          <div className="bg-white p-4 rounded-lg border shadow-sm">
                            <h3 className="font-semibold text-gray-700 mb-3">How to use Files:</h3>
                            <ol className="list-decimal pl-5 text-gray-600 space-y-2 text-sm">
                              <li>Upload your research papers</li>
                              <li>Files will be processed for AI analysis</li>
                              <li>Switch to Chat to ask questions about papers</li>
                              <li>View and manage all files in the library</li>
                            </ol>
                          </div>
                          
                          <div className="bg-white p-4 rounded-lg border shadow-sm">
                            <h3 className="font-semibold text-gray-700 mb-3 flex items-center">
                              <Clock className="h-4 w-4 mr-2 text-blue-500" />
                              Recent Uploads
                            </h3>
                            
                            {recentUploads.length > 0 ? (
                              <div className="space-y-2">
                                {recentUploads.map(file => (
                                  <div key={file.id} className="flex items-center p-2 hover:bg-gray-50 rounded">
                                    <FileText className="h-4 w-4 text-blue-500 mr-2" />
                                    <div className="flex-1">
                                      <p className="text-sm font-medium truncate">{file.name}</p>
                                      <p className="text-xs text-gray-500">{new Date(file.date).toLocaleDateString()}</p>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="text-sm text-gray-500">No recent uploads</p>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent> 
              </Tabs>
            </div>
            
            {/* Footer with Chat Input */}
            <div className="flex-shrink-0 border-t bg-white p-3">
              <form onSubmit={handleSendMessage} className="flex items-center gap-3">
                <div className="flex-1 relative">
                  <Input
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    disabled={isLoading}
                    className="pr-12 py-5 rounded-xl border-gray-200"
                  />
                  <Button 
                    type="submit" 
                    disabled={isLoading || !input.trim()}
                    className="absolute right-1.5 top-1/2 transform -translate-y-1/2 rounded-lg h-9 w-10 bg-gradient-to-r from-blue-500 to-blue-600 text-white"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={clearSession}
                    className="text-red-500 hover:text-red-600 hover:bg-red-50 text-xs font-medium"
                  >
                    Clear
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={downloadChatHistory}
                    className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 text-xs font-medium flex items-center gap-1"
                  >
                    <Download className="h-4 w-4" />
                    Save
                  </Button>
                </div>
              </form>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={{ 
              type: "spring",
              stiffness: 260,
              damping: 20 
            }}
          >
            <Button
              onClick={() => setIsOpen(true)}
              className="group relative rounded-full h-16 w-16 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-lg hover:shadow-xl transition-all duration-300 p-0 flex flex-col items-center justify-center gap-1"
            >
              <div className="absolute inset-0 rounded-full bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
              <Bot className="h-6 w-6" />
              <span className="text-xs font-bold">Ask AI</span>
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AiAssistant;