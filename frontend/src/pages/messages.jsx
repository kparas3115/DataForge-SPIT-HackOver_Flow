import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "../components/ui/button";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardFooter,
} from "../components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { Input } from "../components/ui/input";
import { ScrollArea } from "../components/ui/scroll-area";
import { Alert, AlertDescription } from "../components/ui/alert";
import { User, Send, Loader2, ArrowLeft } from "lucide-react";

const Messages = () => {
    const { userId } = useParams(); // Get the target user ID from URL
    const [targetUser, setTargetUser] = useState(null);
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [message, setMessage] = useState("");
    const [sending, setSending] = useState(false);
    const [messages, setMessages] = useState([]);
    const messagesEndRef = useRef(null);
    const scrollAreaRef = useRef(null);
    const navigate = useNavigate();
    const token = localStorage.getItem("token");
    const API_BASE_URL = "http://localhost:3000/api/network";

    const fetchUserById = async () => {
        try {
            const response = await axios.get(
                `${API_BASE_URL}/users/${userId}`,
                {
                    headers: { Authorization: `Bearer ${token}` },
                }
            );
            setTargetUser(response.data.user);
        } catch (error) {
            console.error("Error fetching user:", error);
            setError("Failed to load user data.");
            setLoading(false);
        }
    };

    const fetchCurrentUser = async () => {
        try {
            const response = await axios.get(
                "http://localhost:3000/api/v1/users/me",
                {
                    headers: { Authorization: `Bearer ${token}` },
                }
            );
            setCurrentUser(response.data.user);

            // Check if users are connected
            const isConnected =
                response.data.user?.connections?.includes(userId);
            if (!isConnected) {
                setError(
                    "You need to connect with this user before messaging them."
                );
            }

            setLoading(false);
        } catch (error) {
            console.error("Error fetching current user:", error);
            setError("Failed to load your profile data.");
            setLoading(false);
        }
    };

    // Load dummy messages for demonstration
    const loadDummyMessages = () => {
        // For demonstration purposes, we'll create some fake messages
        const now = new Date();
        const dummyMessages = [
            {
                id: 1,
                sender: currentUser?._id,
                recipient: userId,
                content:
                    "Hello, I'm interested in your research on neural networks.",
                timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000),
            },
            {
                id: 2,
                sender: userId,
                recipient: currentUser?._id,
                content:
                    "Hi! Thanks for reaching out. I'd be happy to discuss my work with you.",
                timestamp: new Date(now.getTime() - 1.5 * 60 * 60 * 1000),
            },
            {
                id: 3,
                sender: currentUser?._id,
                recipient: userId,
                content:
                    "Great! I'm working on a similar project at my institution and would love to collaborate.",
                timestamp: new Date(now.getTime() - 1 * 60 * 60 * 1000),
            },
            {
                id: 4,
                sender: userId,
                recipient: currentUser?._id,
                content:
                    "That sounds interesting! Let's set up a meeting to discuss the details. What time works for you?",
                timestamp: new Date(now.getTime() - 30 * 60 * 1000),
            },
        ];

        setMessages(dummyMessages);
    };

    useEffect(() => {
        if (!token) {
            navigate("/login");
        } else {
            const loadData = async () => {
                await Promise.all([fetchUserById(), fetchCurrentUser()]);
                // After loading user data, load messages
                loadDummyMessages();
            };
            loadData();
        }
    }, [token, userId, navigate]);

    // Scroll to bottom of messages when new messages arrive
    useEffect(() => {
        if (scrollAreaRef.current) {
            scrollAreaRef.current.scrollTop =
                scrollAreaRef.current.scrollHeight;
        }
    }, [messages]);

    const getInitials = (firstName, lastName) => {
        return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase();
    };

    const handleBack = () => {
        navigate("/my-connections");
    };

    const formatTime = (date) => {
        return date.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const formatDate = (date) => {
        return date.toLocaleDateString();
    };

    const handleSendMessage = async () => {
        if (!message.trim()) return;

        setSending(true);

        try {
            // For demonstration, we'll just add the message locally
            // In a real app, you would send this to the server
            const newMessage = {
                id: Date.now(),
                sender: currentUser?._id,
                recipient: userId,
                content: message,
                timestamp: new Date(),
            };

            setMessages([...messages, newMessage]);
            setMessage("");

            // Simulate API call
            await new Promise((resolve) => setTimeout(resolve, 500));

            // Show success notification
            const successAlert = document.createElement("div");
            successAlert.className =
                "fixed bottom-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded shadow-md z-50";
            successAlert.innerHTML = "<strong>Message Sent!</strong>";
            document.body.appendChild(successAlert);
            setTimeout(() => {
                successAlert.remove();
            }, 2000);
        } catch (error) {
            console.error("Error sending message:", error);
            setError("Failed to send message. Please try again.");
        } finally {
            setSending(false);
        }
    };

    if (loading) {
        return (
            <div className="container mx-auto p-4 flex items-center justify-center min-h-screen">
                <p className="text-gray-600 text-lg flex items-center">
                    <Loader2 className="h-5 w-5 mr-2 animate-spin" /> Loading...
                </p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto p-4">
                <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
                <Button onClick={handleBack} className="mt-4">
                    Back to Connections
                </Button>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-6 min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
            <div className="flex items-center justify-between mb-4">
                <Button
                    variant="outline"
                    className="flex items-center gap-2 text-gray-700 border-gray-300 hover:bg-gray-200"
                    onClick={handleBack}
                >
                    <ArrowLeft className="h-4 w-4" /> Back
                </Button>
                <h1 className="text-2xl font-bold text-gray-800">
                    Conversation with {targetUser?.firstName}{" "}
                    {targetUser?.lastName}
                </h1>
            </div>

            <Card className="bg-white shadow-sm border border-gray-200 h-[calc(100vh-180px)] flex flex-col">
                <CardHeader className="flex flex-row items-center gap-4 border-b border-gray-200 p-4">
                    <Avatar className="h-10 w-10 border-2 border-white shadow-sm">
                        <AvatarImage
                            src={targetUser?.profileImage}
                            alt={`${targetUser?.firstName} ${targetUser?.lastName}`}
                        />
                        <AvatarFallback className="bg-blue-600 text-white">
                            {getInitials(
                                targetUser?.firstName,
                                targetUser?.lastName
                            )}
                        </AvatarFallback>
                    </Avatar>
                    <div>
                        <CardTitle className="text-lg font-semibold text-gray-900">
                            {targetUser?.firstName} {targetUser?.lastName}
                        </CardTitle>
                        <p className="text-sm text-gray-500">
                            {targetUser?.instituteName}
                        </p>
                    </div>
                </CardHeader>

                <div className="flex-grow overflow-hidden" ref={scrollAreaRef}>
                    <ScrollArea className="h-full p-4">
                        <div className="flex flex-col space-y-4">
                            {messages.map((msg) => (
                                <div
                                    key={msg.id}
                                    className={`flex ${
                                        msg.sender === currentUser?._id
                                            ? "justify-end"
                                            : "justify-start"
                                    }`}
                                >
                                    <div
                                        className={`max-w-[80%] rounded-lg p-3 ${
                                            msg.sender === currentUser?._id
                                                ? "bg-blue-600 text-white rounded-br-none"
                                                : "bg-gray-100 text-gray-800 rounded-bl-none"
                                        }`}
                                    >
                                        <p>{msg.content}</p>
                                        <p
                                            className={`text-xs mt-1 text-right ${
                                                msg.sender === currentUser?._id
                                                    ? "text-blue-100"
                                                    : "text-gray-500"
                                            }`}
                                        >
                                            {formatTime(msg.timestamp)}
                                        </p>
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>
                    </ScrollArea>
                </div>

                <CardFooter className="border-t border-gray-200 p-4">
                    <div className="flex w-full gap-2">
                        <Input
                            placeholder="Type your message..."
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyDown={(e) =>
                                e.key === "Enter" &&
                                !e.shiftKey &&
                                handleSendMessage()
                            }
                            className="flex-grow"
                        />
                        <Button
                            type="submit"
                            onClick={handleSendMessage}
                            disabled={sending || !message.trim()}
                            className="bg-blue-600 hover:bg-blue-700 text-white"
                        >
                            {sending ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                                <>
                                    <Send className="h-4 w-4 mr-2" /> Send
                                </>
                            )}
                        </Button>
                    </div>
                </CardFooter>
            </Card>
        </div>
    );
};

export default Messages;
