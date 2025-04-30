// pages/news.jsx
import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
    CalendarIcon,
    ExternalLinkIcon,
    ArrowUpRightIcon,
    BookOpenIcon,
    TagIcon,
} from "lucide-react";

export default function NewsPage() {
    const [news, setNews] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeCategory, setActiveCategory] = useState("all");

    const categories = [
        {
            value: "all",
            label: "All News",
            icon: <BookOpenIcon className="h-4 w-4 mr-1" />,
        },
        {
            value: "science",
            label: "Science",
            icon: <TagIcon className="h-4 w-4 mr-1" />,
        },
        {
            value: "technology",
            label: "Technology",
            icon: <TagIcon className="h-4 w-4 mr-1" />,
        },
        {
            value: "computer science",
            label: "Computer Science",
            icon: <TagIcon className="h-4 w-4 mr-1" />,
        },
    ];

    // Function to determine background gradient based on category
    const getCategoryGradient = (category) => {
        switch (category.toLowerCase()) {
            case "science":
                return "bg-gradient-to-r from-blue-600 to-blue-400";
            case "technology":
                return "bg-gradient-to-r from-indigo-600 to-purple-400";
            case "computer science":
                return "bg-gradient-to-r from-cyan-600 to-blue-400";
            default:
                return "bg-gradient-to-r from-blue-500 to-indigo-500";
        }
    };

    // Function to determine text color based on category
    const getCategoryTextColor = (category) => {
        switch (category.toLowerCase()) {
            case "science":
                return "text-blue-600";
            case "technology":
                return "text-indigo-600";
            case "computer science":
                return "text-cyan-600";
            default:
                return "text-blue-500";
        }
    };

    useEffect(() => {
        const fetchNews = async () => {
            setLoading(true);
            try {
                // Get the JWT token from localStorage
                const token = localStorage.getItem("token");

                if (!token) {
                    setError("Authentication token not found. Please log in.");
                    setLoading(false);
                    return;
                }

                const url =
                    activeCategory === "all"
                        ? "http://localhost:3000/api/news/"
                        : `http://localhost:3000/api/news/?category=${encodeURIComponent(
                              activeCategory
                          )}`;

                const response = await fetch(url, {
                    headers: {
                        Authorization: `Bearer ${token}`,
                        "Content-Type": "application/json",
                    },
                });

                // Handle unauthorized responses
                if (response.status === 401) {
                    setError("Authentication failed. Please log in again.");
                    setLoading(false);
                    return;
                }

                const data = await response.json();

                if (data.news) {
                    setNews(data.news);
                } else if (data.success === false) {
                    setError(data.message || "Failed to fetch news");
                } else {
                    // If the response is the array directly or has a different structure
                    setNews(Array.isArray(data) ? data : data.news || []);
                }
            } catch (err) {
                setError("Error connecting to the news service");
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchNews();
    }, [activeCategory]);

    // Format date to be more readable
    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return new Intl.DateTimeFormat("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
        }).format(date);
    };

    // Truncate content to avoid very long descriptions
    const truncateContent = (content, maxLength = 150) => {
        if (!content) return "";
        if (content.length <= maxLength) return content;
        return content.substring(0, maxLength) + "...";
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-50 to-blue-50">
            <div className="container mx-auto py-12 px-4">
                <div className="flex flex-col space-y-8">
                    <div className="text-center space-y-3 mb-6">
                        <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                            Research News
                        </h1>
                        <p className="text-slate-600 text-lg max-w-2xl mx-auto">
                            Discover the latest research updates from top
                            scientific and technology sources around the world
                        </p>
                        <div className="w-24 h-1 bg-gradient-to-r from-blue-400 to-indigo-500 mx-auto mt-2 rounded-full"></div>
                    </div>

                    <div className="bg-white rounded-xl shadow-xl p-6 border border-blue-100">
                        <Tabs
                            defaultValue="all"
                            value={activeCategory}
                            onValueChange={setActiveCategory}
                            className="w-full"
                        >
                            <TabsList className="mb-6 bg-blue-50 p-1 rounded-lg border border-blue-100 w-full flex justify-between">
                                {categories.map((category) => (
                                    <TabsTrigger
                                        key={category.value}
                                        value={category.value}
                                        className="flex items-center px-6 py-2 rounded-md font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-blue-600 data-[state=active]:shadow-sm data-[state=active]:font-semibold"
                                    >
                                        {category.icon}
                                        {category.label}
                                    </TabsTrigger>
                                ))}
                            </TabsList>

                            {categories.map((category) => (
                                <TabsContent
                                    key={category.value}
                                    value={category.value}
                                    className="space-y-6"
                                >
                                    {loading ? (
                                        // Enhanced loading skeletons
                                        <div className="grid grid-cols-1 gap-6">
                                            {Array(3)
                                                .fill()
                                                .map((_, i) => (
                                                    <Card
                                                        key={i}
                                                        className="w-full overflow-hidden border-0 shadow-md bg-white"
                                                    >
                                                        <div className="h-2 bg-gradient-to-r from-blue-400 to-indigo-500"></div>
                                                        <CardHeader>
                                                            <Skeleton className="h-6 w-3/4 mb-2 bg-blue-100" />
                                                            <Skeleton className="h-4 w-1/4 bg-blue-100" />
                                                        </CardHeader>
                                                        <CardContent>
                                                            <Skeleton className="h-4 w-full mb-2 bg-blue-100" />
                                                            <Skeleton className="h-4 w-full mb-2 bg-blue-100" />
                                                            <Skeleton className="h-4 w-2/3 bg-blue-100" />
                                                        </CardContent>
                                                    </Card>
                                                ))}
                                        </div>
                                    ) : error ? (
                                        // Enhanced error message
                                        <Card className="w-full border-0 shadow-lg bg-white overflow-hidden">
                                            <div className="h-2 bg-red-500"></div>
                                            <CardHeader>
                                                <CardTitle className="text-red-500 font-bold">
                                                    Error Loading News
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent>
                                                <p className="text-slate-600">
                                                    {error}
                                                </p>
                                            </CardContent>
                                        </Card>
                                    ) : news.length === 0 ? (
                                        // Enhanced no news found message
                                        <Card className="w-full border-0 shadow-lg bg-white overflow-hidden">
                                            <div className="h-2 bg-blue-500"></div>
                                            <CardHeader>
                                                <CardTitle className="text-blue-600">
                                                    No News Available
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent>
                                                <p className="text-slate-600">
                                                    No research news articles
                                                    found for this category.
                                                </p>
                                            </CardContent>
                                        </Card>
                                    ) : (
                                        // Enhanced news articles display
                                        <ScrollArea className="h-[700px] pr-4">
                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                                {news.map((item) => (
                                                    <Card
                                                        key={item.guid}
                                                        className="w-full border-0 shadow-md hover:shadow-lg transition-all duration-300 bg-white overflow-hidden group"
                                                    >
                                                        <div
                                                            className={`h-2 ${getCategoryGradient(
                                                                item.category
                                                            )}`}
                                                        ></div>
                                                        <CardHeader className="pb-2">
                                                            <div className="flex justify-between items-start">
                                                                <div>
                                                                    <CardTitle className="text-xl text-slate-800 group-hover:text-blue-600 transition-colors duration-300">
                                                                        <a
                                                                            href={
                                                                                item.link
                                                                            }
                                                                            target="_blank"
                                                                            rel="noopener noreferrer"
                                                                            className="hover:underline flex items-center"
                                                                        >
                                                                            {
                                                                                item.title
                                                                            }
                                                                            <ArrowUpRightIcon className="ml-2 h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                                                                        </a>
                                                                    </CardTitle>
                                                                    <CardDescription className="mt-2 flex items-center text-sm space-x-4 text-slate-500">
                                                                        <span className="flex items-center">
                                                                            <CalendarIcon className="mr-1 h-4 w-4" />
                                                                            {formatDate(
                                                                                item.pubDate
                                                                            )}
                                                                        </span>
                                                                        <span className="font-medium">
                                                                            {
                                                                                item.source
                                                                            }
                                                                        </span>
                                                                    </CardDescription>
                                                                </div>
                                                                <Badge
                                                                    variant="outline"
                                                                    className={`${getCategoryTextColor(
                                                                        item.category
                                                                    )} border-current font-medium`}
                                                                >
                                                                    {
                                                                        item.category
                                                                    }
                                                                </Badge>
                                                            </div>
                                                        </CardHeader>
                                                        <CardContent>
                                                            <p className="text-sm text-slate-600 leading-relaxed">
                                                                {truncateContent(
                                                                    item.content
                                                                )}
                                                            </p>
                                                        </CardContent>
                                                        <CardFooter className="pt-0">
                                                            <a
                                                                href={item.link}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="text-sm text-blue-500 hover:text-blue-700 font-medium hover:underline flex items-center group-hover:translate-x-1 transition-transform duration-300"
                                                            >
                                                                Read full
                                                                article
                                                                <ArrowUpRightIcon className="ml-1 h-4 w-4" />
                                                            </a>
                                                        </CardFooter>
                                                    </Card>
                                                ))}
                                            </div>
                                        </ScrollArea>
                                    )}
                                </TabsContent>
                            ))}
                        </Tabs>
                    </div>
                </div>
            </div>
        </div>
    );
}
