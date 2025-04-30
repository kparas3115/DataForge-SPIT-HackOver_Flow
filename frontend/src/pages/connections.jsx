import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "../components/ui/select";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardFooter,
} from "../components/ui/card";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "../components/ui/dialog";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { Badge } from "../components/ui/badge";
import { ScrollArea } from "../components/ui/scroll-area";
import { Separator } from "../components/ui/separator";
// Using an alert-based approach instead of toast to avoid import issues
import {
    User,
    University,
    BookOpen,
    FileText,
    Calendar,
    ExternalLink,
    CheckCircle,
    Loader2,
} from "lucide-react";

const Connections = () => {
    const [users, setUsers] = useState([]);
    const [meta, setMeta] = useState({
        total: 0,
        limit: 10,
        skip: 0,
        hasMore: false,
    });
    const [sortBy, setSortBy] = useState("firstName");
    const [sortOrder, setSortOrder] = useState(1);
    const [searchQuery, setSearchQuery] = useState("");
    const [instituteName, setInstituteName] = useState("");
    const [interests, setInterests] = useState("");
    const [selectedUser, setSelectedUser] = useState(null);
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [connectionStatus, setConnectionStatus] = useState({
        connecting: false,
        success: false,
        userId: null,
    });

    const API_BASE_URL = "http://localhost:3000/api/network";
    const token = localStorage.getItem("token");
    const navigate = useNavigate();

    const fetchUsers = async (reset = false) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/users`, {
                headers: { Authorization: `Bearer ${token}` },
                params: {
                    limit: meta.limit,
                    skip: reset ? 0 : meta.skip,
                    sortBy,
                    sortOrder,
                },
            });
            const { users: fetchedUsers, meta: fetchedMeta } = response.data;
            setUsers(reset ? fetchedUsers : [...users, ...fetchedUsers]);
            setMeta(fetchedMeta);
        } catch (error) {
            console.error("Error fetching users:", error);
            setError("Failed to load users. Please try again.");
        }
    };

    const searchUsers = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/search`, {
                headers: { Authorization: `Bearer ${token}` },
                params: {
                    query: searchQuery || undefined,
                    instituteName: instituteName || undefined,
                    interests: interests || undefined,
                    limit: meta.limit,
                    skip: 0,
                },
            });
            const { users: fetchedUsers, meta: fetchedMeta } = response.data;
            setUsers(fetchedUsers);
            setMeta(fetchedMeta);
        } catch (error) {
            console.error("Error searching users:", error);
            setError("Failed to search users. Please try again.");
        }
    };

    const fetchUserById = async (userId) => {
        try {
            const response = await axios.get(
                `${API_BASE_URL}/users/${userId}`,
                {
                    headers: { Authorization: `Bearer ${token}` },
                }
            );

            // Check if the user is already connected
            const isConnected = currentUser?.connections?.includes(userId);

            setSelectedUser({
                ...response.data.user,
                isConnected,
            });
        } catch (error) {
            console.error("Error fetching user by ID:", error);
            setError("Failed to load user profile.");
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
        } catch (error) {
            console.error("Error fetching current user:", error);
            setError("Failed to load current user data.");
        }
    };

    const handleConnect = async (userId) => {
        setConnectionStatus({ connecting: true, success: false, userId });

        try {
            const response = await axios.post(
                `${API_BASE_URL}/connect/${userId}`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (response.data.success) {
                setConnectionStatus({
                    connecting: false,
                    success: true,
                    userId,
                });

                // Show a success message
                const successAlert = document.createElement("div");
                successAlert.className =
                    "fixed top-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded shadow-md z-50";
                successAlert.innerHTML =
                    "<strong>Success!</strong> User has been added to your connections!";
                document.body.appendChild(successAlert);
                setTimeout(() => {
                    successAlert.remove();
                }, 3000);

                // Wait for 1.5 seconds before redirecting
                setTimeout(() => {
                    navigate("/my-connections");
                }, 1500);
            }
        } catch (error) {
            console.error(
                "Error connecting user:",
                error.response?.data || error.message
            );
            setConnectionStatus({
                connecting: false,
                success: false,
                userId: null,
            });

            const errorAlert = document.createElement("div");
            errorAlert.className =
                "fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-md z-50";
            errorAlert.innerHTML =
                "<strong>Error!</strong> Failed to connect with this user. Please try again.";
            document.body.appendChild(errorAlert);
            setTimeout(() => {
                errorAlert.remove();
            }, 3000);
        }
    };

    const handleMessage = (userId) => {
        // Check if connected first
        if (!currentUser?.connections?.includes(userId)) {
            const warningAlert = document.createElement("div");
            warningAlert.className =
                "fixed top-4 right-4 bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded shadow-md z-50";
            warningAlert.innerHTML =
                "<strong>Warning!</strong> You need to connect with this user first before messaging them.";
            document.body.appendChild(warningAlert);
            setTimeout(() => {
                warningAlert.remove();
            }, 3000);
            return;
        }

        navigate(`/messages/${userId}`);
    };

    useEffect(() => {
        if (!token) {
            navigate("/login");
            return;
        }
        const loadData = async () => {
            setLoading(true);
            await Promise.all([fetchUsers(true), fetchCurrentUser()]);
            setLoading(false);
        };
        loadData();
    }, [sortBy, sortOrder, token, navigate]);

    const handleSearch = (e) => {
        e.preventDefault();
        searchUsers();
    };

    const loadMore = () => {
        fetchUsers();
    };

    const getInitials = (firstName, lastName) => {
        return `${firstName?.[0] || ""}${lastName?.[0] || ""}`.toUpperCase();
    };

    const getPublicationsCount = (papers) => {
        return papers?.length || 0;
    };

    const handleViewConnections = () => {
        navigate("/my-connections");
    };

    const isUserConnected = (userId) => {
        return currentUser?.connections?.includes(userId);
    };

    if (loading) {
        return (
            <div className="container mx-auto p-4 flex items-center justify-center min-h-screen">
                <p className="text-gray-600 text-lg">Loading...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container mx-auto p-4">
                <p className="text-red-500">{error}</p>
                <Button
                    onClick={() => window.location.reload()}
                    className="mt-4"
                >
                    Retry
                </Button>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-4">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold">Networking</h1>
                <Button
                    onClick={handleViewConnections}
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                    My Connections
                </Button>
            </div>

            <form
                onSubmit={handleSearch}
                className="flex flex-col sm:flex-row gap-4 mb-6"
            >
                <Input
                    placeholder="Search by name or bio..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1"
                />
                <Input
                    placeholder="Filter by institute..."
                    value={instituteName}
                    onChange={(e) => setInstituteName(e.target.value)}
                    className="flex-1"
                />
                <Input
                    placeholder="Filter by interests..."
                    value={interests}
                    onChange={(e) => setInterests(e.target.value)}
                    className="flex-1"
                />
                <Button type="submit">Search</Button>
            </form>

            <div className="flex flex-col sm:flex-row gap-4 mb-6">
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Sort by:</span>
                    <Select value={sortBy} onValueChange={setSortBy}>
                        <SelectTrigger className="w-[180px]">
                            <SelectValue placeholder="Sort by" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="firstName">
                                First Name
                            </SelectItem>
                            <SelectItem value="lastName">Last Name</SelectItem>
                            <SelectItem value="instituteName">
                                Institute Name
                            </SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Order:</span>
                    <Select
                        value={sortOrder.toString()}
                        onValueChange={(val) => setSortOrder(Number(val))}
                    >
                        <SelectTrigger className="w-[120px]">
                            <SelectValue placeholder="Order" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="1">Ascending</SelectItem>
                            <SelectItem value="-1">Descending</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {users.map((user) => (
                    <Card
                        key={user._id}
                        className="overflow-hidden border border-gray-200 hover:shadow-md transition-shadow"
                    >
                        <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200 pb-3">
                            <div className="flex items-center gap-3 justify-between">
                                <div className="flex items-center gap-3">
                                    <Avatar className="h-12 w-12 border-2 border-white shadow-sm">
                                        <AvatarImage
                                            src={user.profileImage}
                                            alt={`${user.firstName} ${user.lastName}`}
                                        />
                                        <AvatarFallback className="bg-blue-600 text-white">
                                            {getInitials(
                                                user.firstName,
                                                user.lastName
                                            )}
                                        </AvatarFallback>
                                    </Avatar>
                                    <div>
                                        <CardTitle className="text-lg font-bold text-gray-900">
                                            {user.firstName} {user.lastName}
                                        </CardTitle>
                                        <div className="text-sm text-blue-600 font-medium flex items-center mt-1">
                                            <University className="h-3 w-3 mr-1" />
                                            {user.instituteName}
                                        </div>
                                    </div>
                                </div>
                                {isUserConnected(user._id) && (
                                    <Badge className="bg-green-100 text-green-800 flex items-center gap-1">
                                        <CheckCircle className="h-3 w-3" />{" "}
                                        Connected
                                    </Badge>
                                )}
                            </div>
                        </CardHeader>
                        <CardContent className="p-4">
                            <div className="flex items-center mb-3 text-sm font-medium text-gray-700">
                                <FileText className="h-4 w-4 mr-1.5 text-blue-600" />
                                <span>
                                    {getPublicationsCount(user.papers)}{" "}
                                    publications
                                </span>
                            </div>
                            <div className="mb-2">
                                <h4 className="text-sm font-medium text-gray-700 mb-1.5 flex items-center">
                                    <BookOpen className="h-4 w-4 mr-1.5 text-blue-600" />
                                    Research Interests
                                </h4>
                                <div className="flex flex-wrap gap-1.5">
                                    {user.interests
                                        ?.split(",")
                                        .slice(0, 3)
                                        .map((interest, idx) => (
                                            <Badge
                                                key={idx}
                                                className="bg-blue-100 text-blue-800 border-blue-200 hover:bg-blue-200 text-xs px-2 py-0.5 rounded-full"
                                            >
                                                {interest.trim()}
                                            </Badge>
                                        ))}
                                    {user.interests?.split(",").length > 3 && (
                                        <Badge className="bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200 text-xs px-2 py-0.5 rounded-full">
                                            +
                                            {user.interests.split(",").length -
                                                3}{" "}
                                            more
                                        </Badge>
                                    )}
                                </div>
                            </div>
                        </CardContent>
                        <CardFooter className="bg-gray-50 border-t border-gray-200 p-3 flex justify-end gap-2">
                            {isUserConnected(user._id) ? (
                                <>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="text-blue-700 border-blue-200 bg-blue-50 hover:bg-blue-100"
                                        onClick={() => handleMessage(user._id)}
                                    >
                                        Message
                                    </Button>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="text-blue-700 border-blue-200 bg-blue-50 hover:bg-blue-100"
                                        onClick={() => fetchUserById(user._id)}
                                    >
                                        View Profile
                                    </Button>
                                </>
                            ) : (
                                <>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="text-blue-700 border-blue-200 bg-blue-50 hover:bg-blue-100"
                                        onClick={() => fetchUserById(user._id)}
                                    >
                                        View Profile
                                    </Button>
                                    <Button
                                        size="sm"
                                        className="bg-blue-600 hover:bg-blue-700 text-white"
                                        onClick={() => handleConnect(user._id)}
                                        disabled={
                                            connectionStatus.connecting &&
                                            connectionStatus.userId === user._id
                                        }
                                    >
                                        {connectionStatus.connecting &&
                                        connectionStatus.userId === user._id ? (
                                            <>
                                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />{" "}
                                                Connecting...
                                            </>
                                        ) : connectionStatus.success &&
                                          connectionStatus.userId ===
                                              user._id ? (
                                            <>
                                                <CheckCircle className="h-4 w-4 mr-1" />{" "}
                                                Connected
                                            </>
                                        ) : (
                                            "Connect"
                                        )}
                                    </Button>
                                </>
                            )}
                        </CardFooter>
                    </Card>
                ))}
            </div>

            {meta.hasMore && (
                <Button
                    onClick={loadMore}
                    className="mt-6 w-full sm:w-auto mx-auto block"
                >
                    Load More
                </Button>
            )}

            <Dialog
                open={!!selectedUser}
                onOpenChange={(open) => !open && setSelectedUser(null)}
            >
                {selectedUser && (
                    <DialogContent className="sm:max-w-md md:max-w-xl lg:max-w-2xl bg-gradient-to-br from-slate-50 via-white to-blue-50 p-0 overflow-hidden rounded-lg border border-gray-200">
                        <DialogHeader className="px-6 pt-6 pb-4 bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
                            <div className="flex items-center gap-4 justify-between">
                                <div className="flex items-center gap-4">
                                    <Avatar className="h-16 w-16 border-2 border-white shadow-sm">
                                        <AvatarImage
                                            src={selectedUser.profileImage}
                                            alt={`${selectedUser.firstName} ${selectedUser.lastName}`}
                                        />
                                        <AvatarFallback className="text-lg bg-blue-600 text-white">
                                            {getInitials(
                                                selectedUser.firstName,
                                                selectedUser.lastName
                                            )}
                                        </AvatarFallback>
                                    </Avatar>
                                    <div>
                                        <DialogTitle className="text-2xl font-bold text-gray-900">
                                            {selectedUser.firstName}{" "}
                                            {selectedUser.lastName}
                                        </DialogTitle>
                                        <DialogDescription className="text-base text-blue-600 font-medium mt-1 flex items-center">
                                            <University className="h-4 w-4 mr-1.5" />
                                            {selectedUser.instituteName}
                                        </DialogDescription>
                                    </div>
                                </div>
                                {selectedUser.isConnected && (
                                    <Badge className="bg-green-100 text-green-800 flex items-center gap-1 px-3 py-1">
                                        <CheckCircle className="h-4 w-4 mr-1" />{" "}
                                        Connected
                                    </Badge>
                                )}
                            </div>
                        </DialogHeader>
                        <ScrollArea className="max-h-[65vh] px-6 py-4">
                            <div className="space-y-5">
                                {selectedUser.shortBio && (
                                    <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                                        <h3 className="text-lg font-semibold mb-2 text-gray-800 flex items-center">
                                            <User className="h-4 w-4 mr-2 text-blue-600" />
                                            About
                                        </h3>
                                        <p className="text-gray-600">
                                            {selectedUser.shortBio}
                                        </p>
                                    </div>
                                )}
                                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                                    <h3 className="text-lg font-semibold mb-2 text-gray-800 flex items-center">
                                        <BookOpen className="h-4 w-4 mr-2 text-blue-600" />
                                        Research Interests
                                    </h3>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedUser.interests
                                            .split(",")
                                            .map((interest, idx) => (
                                                <Badge
                                                    key={idx}
                                                    className="bg-blue-100 text-blue-800 border-blue-200 hover:bg-blue-200 text-sm px-3 py-1 rounded-full"
                                                >
                                                    {interest.trim()}
                                                </Badge>
                                            ))}
                                    </div>
                                </div>
                                {selectedUser.papers &&
                                    selectedUser.papers.length > 0 && (
                                        <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                                            <h3 className="text-lg font-semibold mb-2 text-gray-800 flex items-center">
                                                <FileText className="h-4 w-4 mr-2 text-blue-600" />
                                                Publications
                                            </h3>
                                            <ul className="space-y-2">
                                                {selectedUser.papers.map(
                                                    (paper, index) => (
                                                        <li
                                                            key={index}
                                                            className="py-2 px-3 bg-blue-50 rounded-md flex items-center"
                                                        >
                                                            <span className="h-5 w-5 flex items-center justify-center bg-blue-100 text-blue-600 rounded-full mr-2 text-xs font-medium">
                                                                {index + 1}
                                                            </span>
                                                            <a
                                                                href={paper.url}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="text-blue-600 hover:underline text-sm font-medium flex-1"
                                                            >
                                                                {paper.name}
                                                            </a>
                                                            <ExternalLink className="h-3.5 w-3.5 text-blue-400 ml-1 flex-shrink-0" />
                                                        </li>
                                                    )
                                                )}
                                            </ul>
                                        </div>
                                    )}
                                {selectedUser.createdAt && (
                                    <div className="mt-4 text-sm text-gray-500 flex items-center justify-end">
                                        <Calendar className="h-3.5 w-3.5 mr-1.5" />
                                        Active since:{" "}
                                        {new Date(
                                            selectedUser.createdAt
                                        ).toLocaleDateString()}
                                    </div>
                                )}
                            </div>
                        </ScrollArea>
                        <DialogFooter className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex sm:justify-between gap-4">
                            <Button
                                variant="outline"
                                onClick={() => setSelectedUser(null)}
                                className="border-gray-300 text-gray-700"
                            >
                                Close
                            </Button>
                            <div className="flex gap-2">
                                {selectedUser.isConnected ? (
                                    <Button
                                        className="bg-blue-600 hover:bg-blue-700 text-white"
                                        onClick={() =>
                                            handleMessage(selectedUser._id)
                                        }
                                    >
                                        Message
                                    </Button>
                                ) : (
                                    <Button
                                        className="bg-blue-600 hover:bg-blue-700 text-white"
                                        onClick={() =>
                                            handleConnect(selectedUser._id)
                                        }
                                        disabled={connectionStatus.connecting}
                                    >
                                        {connectionStatus.connecting &&
                                        connectionStatus.userId ===
                                            selectedUser._id ? (
                                            <>
                                                <Loader2 className="h-4 w-4 mr-1 animate-spin" />{" "}
                                                Connecting...
                                            </>
                                        ) : (
                                            "Connect"
                                        )}
                                    </Button>
                                )}
                            </div>
                        </DialogFooter>
                    </DialogContent>
                )}
            </Dialog>
        </div>
    );
};

export default Connections;
