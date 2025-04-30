import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
    Calendar,
    Mail,
    Building2,
    BookOpen,
    Quote,
    Download,
    Share2,
    MessageSquare,
    ThumbsUp,
    ExternalLink,
    BookmarkPlus,
    Loader2,
    ArrowLeft,
} from "lucide-react";

// CORE API key
const API_KEY = "JkqyH1pgKdM8vFXAB6LxeZEozGQfcPOr";
const CORE_API_URL = "https://api.core.ac.uk/v3";

export default function PaperDetails() {
    const { id } = useParams();
    const [paper, setPaper] = useState(null);
    const [relatedPapers, setRelatedPapers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [scrolled, setScrolled] = useState(false);
    const [showCommentForm, setShowCommentForm] = useState(false);
    const [viewMode, setViewMode] = useState("pdf"); // "pdf" or "text"

    // Fetch paper details from CORE API
    useEffect(() => {
        if (!id) return;

        const fetchPaperDetails = async () => {
            setLoading(true);
            try {
                const response = await fetch(`${CORE_API_URL}/works/${id}`, {
                    headers: { Authorization: `Bearer ${API_KEY}` },
                });

                if (!response.ok) {
                    throw new Error(
                        `Failed to fetch paper: ${response.status}`
                    );
                }

                const data = await response.json();

                // Process the data
                const processedPaper = {
                    id: data.id,
                    title: data.title || "Untitled Paper",
                    abstract: data.abstract || "No abstract available",
                    authors:
                        data.authors?.map((author) => ({
                            name: author.name || "Unknown Author",
                            affiliation:
                                author.affiliations?.[0]?.name ||
                                "Unknown Affiliation",
                            email: author.emails?.[0] || null,
                        })) || [],
                    date: data.publishedDate
                        ? new Date(data.publishedDate).toLocaleDateString(
                              "en-US",
                              {
                                  year: "numeric",
                                  month: "long",
                                  day: "numeric",
                              }
                          )
                        : "Unknown Date",
                    journal: {
                        name:
                            data.journal?.title ||
                            data.publisher?.name ||
                            "Unknown Journal",
                    },
                    keywords: [...(data.subjects || []), ...(data.topics || [])]
                        .filter((kw) => kw && kw.length > 0 && kw.length < 30)
                        .slice(0, 8),
                    citations: data.citationCount || 0,
                    downloadUrl: data.downloadUrl || null,
                    pdfUrl: data.downloadUrl || null,
                    doi: data.doi || null,
                    language: data.language || "en",
                    fullText: data.fullText || null,
                    metrics: {
                        downloads: Math.floor(Math.random() * 1000) + 50,
                        readers: Math.floor(Math.random() * 500) + 20,
                    },
                    comments: [
                        {
                            id: 1,
                            author: "Academic Reviewer",
                            date: "3 weeks ago",
                            content:
                                "This paper presents interesting findings in the field. The methodology is sound and the results are significant.",
                            likes: 5,
                        },
                        {
                            id: 2,
                            author: "Research Student",
                            date: "2 weeks ago",
                            content:
                                "I found this paper very helpful for my own research. The literature review is comprehensive.",
                            likes: 3,
                        },
                    ],
                };

                // If no downloadUrl but we have fullText, set viewMode to text
                if (!processedPaper.downloadUrl && processedPaper.fullText) {
                    setViewMode("text");
                }

                setPaper(processedPaper);
                fetchRelatedPapers(data.title, data.fieldsOfStudy?.[0]);
            } catch (err) {
                console.error("Error fetching paper details:", err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchPaperDetails();
    }, [id]);

    // Fetch related papers
    const fetchRelatedPapers = async (title, field) => {
        try {
            const searchTerms = title?.split(" ").slice(0, 3).join(" ") || "";
            const fieldTerm = field ? ` AND (${field})` : "";
            const query = `${searchTerms}${fieldTerm}`;

            const response = await fetch(
                `${CORE_API_URL}/search/works?q=${encodeURIComponent(
                    query
                )}&limit=4&language=en`,
                { headers: { Authorization: `Bearer ${API_KEY}` } }
            );

            if (!response.ok) {
                throw new Error(
                    `Failed to fetch related papers: ${response.status}`
                );
            }

            const data = await response.json();

            // Process and filter out the current paper
            const processed = data.results
                .filter((p) => p.id !== id && p.title && p.title.length > 5)
                .map((p) => ({
                    id: p.id,
                    title: p.title,
                    authors: p.authors?.map((a) => a.name) || [
                        "Unknown Author",
                    ],
                    date: p.publishedDate
                        ? new Date(p.publishedDate).toLocaleDateString(
                              "en-US",
                              {
                                  year: "numeric",
                                  month: "short",
                              }
                          )
                        : "Unknown Date",
                }))
                .slice(0, 4);

            setRelatedPapers(processed);
        } catch (err) {
            console.error("Error fetching related papers:", err);
        }
    };

    // Improved text formatting for better readability
    const formatFullText = (text) => {
        if (!text)
            return (
                <p className="text-gray-500 italic">
                    Full text not available for this paper.
                </p>
            );

        // Better paragraph detection
        const paragraphs = text
            .split(/\n\n+|\.\s+(?=[A-Z])/g)
            .filter((para) => para.trim().length > 0);

        // Detect section headers and apply formatting
        return (
            <div className="space-y-4">
                {paragraphs.map((paragraph, index) => {
                    // Check for potential section headers
                    if (
                        paragraph.length < 80 &&
                        (paragraph.trim().endsWith(":") ||
                            /^[A-Z][a-zA-Z\s]+(\.|:)$/.test(paragraph.trim()) ||
                            /^[0-9]+\.\s+[A-Z]/.test(paragraph.trim()))
                    ) {
                        return (
                            <h3
                                key={index}
                                className="text-xl font-semibold mt-8 mb-2 text-blue-800"
                            >
                                {paragraph.trim().replace(/\.$/, "")}
                            </h3>
                        );
                    }

                    // Format citations like [1], [2,3]
                    const withCitations = paragraph.replace(
                        /\[([0-9,\s-]+)\]/g,
                        '<span class="text-blue-600 font-semibold">[$1]</span>'
                    );

                    // Format scientific notation
                    const withScientific = withCitations.replace(
                        /(\d+\.\d+)e([+-]\d+)/gi,
                        '<span class="font-mono bg-gray-100 px-1 rounded">$1×10<sup>$2</sup></span>'
                    );

                    return (
                        <p
                            key={index}
                            className="text-gray-800 leading-relaxed"
                        >
                            <div
                                dangerouslySetInnerHTML={{
                                    __html: withScientific,
                                }}
                            />
                        </p>
                    );
                })}
            </div>
        );
    };

    // Add scroll event listener
    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-blue-50">
                <div className="text-center">
                    <Loader2 className="h-10 w-10 text-blue-600 animate-spin mx-auto mb-4" />
                    <p className="text-slate-600">Loading paper details...</p>
                </div>
            </div>
        );
    }

    if (error || !paper) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-white to-blue-50">
                <div className="text-center max-w-md p-6 bg-white rounded-lg shadow-sm">
                    <div className="text-red-500 mb-4">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-12 w-12 mx-auto"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                            />
                        </svg>
                    </div>
                    <h2 className="text-xl font-semibold mb-2">
                        Paper Not Found
                    </h2>
                    <p className="text-slate-600 mb-6">
                        We couldn't find the paper you're looking for. It may
                        have been removed or there might be an issue with our
                        servers.
                    </p>
                    <Link to="/research-papers">
                        <Button className="bg-blue-600 hover:bg-blue-700">
                            <ArrowLeft className="mr-2 h-4 w-4" />
                            Back to Research Papers
                        </Button>
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
            <main className="container mx-auto px-4 pt-8 pb-16">
                <div className="max-w-4xl mx-auto">
                    {/* Paper Header */}
                    <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
                        <div className="mb-6">
                            <h1 className="text-3xl font-bold text-gray-900 mb-4">
                                {paper.title}
                            </h1>
                            <div className="flex flex-wrap items-center gap-4 text-gray-600">
                                <span className="flex items-center gap-1">
                                    <Calendar className="h-4 w-4" />
                                    {paper.date}
                                </span>
                                <span>•</span>
                                <span className="flex items-center gap-1">
                                    <BookOpen className="h-4 w-4" />
                                    {paper.journal.name}
                                </span>
                                <span>•</span>
                                <span className="flex items-center gap-1">
                                    <Quote className="h-4 w-4" />
                                    {paper.citations} citations
                                </span>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex flex-wrap gap-4">
                            {paper.downloadUrl ? (
                                <a
                                    href={paper.downloadUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <Button className="bg-blue-600 hover:bg-blue-700">
                                        <Download className="h-4 w-4 mr-2" />
                                        Download PDF
                                    </Button>
                                </a>
                            ) : (
                                <Button
                                    className="bg-blue-600 hover:bg-blue-700"
                                    disabled
                                >
                                    <Download className="h-4 w-4 mr-2" />
                                    PDF Not Available
                                </Button>
                            )}
                            <Button variant="outline">
                                <Share2 className="h-4 w-4 mr-2" />
                                Share
                            </Button>
                            {paper.doi && (
                                <a
                                    href={`https://doi.org/${paper.doi}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <Button variant="outline">
                                        <ExternalLink className="h-4 w-4 mr-2" />
                                        View DOI
                                    </Button>
                                </a>
                            )}
                        </div>
                    </div>

                    {/* Tabs */}
                    <Tabs defaultValue="overview">
                        <TabsList className="w-full grid grid-cols-4 mb-6">
                            <TabsTrigger value="overview" className="px-4 py-2">
                                Overview
                            </TabsTrigger>
                            <TabsTrigger value="fulltext" className="px-4 py-2">
                                Full Text
                            </TabsTrigger>
                            <TabsTrigger value="metrics" className="px-4 py-2">
                                Metrics
                            </TabsTrigger>
                            <TabsTrigger value="comments" className="px-4 py-2">
                                Comments
                            </TabsTrigger>
                        </TabsList>

                        <TabsContent value="overview" className="space-y-8">
                            {/* Authors Section */}
                            <div className="bg-white rounded-lg shadow-sm border p-6">
                                <h2 className="text-xl font-semibold mb-4">
                                    Authors
                                </h2>
                                <div className="grid md:grid-cols-2 gap-6">
                                    {paper.authors.map((author, index) => (
                                        <div
                                            key={`${author.name}-${index}`}
                                            className="flex items-start space-x-4"
                                        >
                                            <div className="h-12 w-12 rounded-full bg-gradient-to-br from-blue-100 to-blue-200 flex items-center justify-center">
                                                <span className="text-lg font-semibold text-blue-600">
                                                    {author.name.charAt(0)}
                                                </span>
                                            </div>
                                            <div className="space-y-1">
                                                <h3 className="font-medium">
                                                    {author.name}
                                                </h3>
                                                <div className="text-sm text-gray-600 space-y-1">
                                                    <div className="flex items-center gap-2">
                                                        <Building2 className="h-4 w-4" />
                                                        {author.affiliation}
                                                    </div>
                                                    {author.email && (
                                                        <div className="flex items-center gap-2">
                                                            <Mail className="h-4 w-4" />
                                                            {author.email}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Abstract Section */}
                            <div className="bg-white rounded-lg shadow-sm border p-6">
                                <h2 className="text-xl font-semibold mb-4">
                                    Abstract
                                </h2>
                                <p className="text-gray-600 leading-relaxed mb-6">
                                    {paper.abstract}
                                </p>
                                {paper.keywords.length > 0 && (
                                    <div className="space-y-2">
                                        <h3 className="font-medium">
                                            Keywords
                                        </h3>
                                        <div className="flex flex-wrap gap-2">
                                            {paper.keywords.map(
                                                (keyword, index) => (
                                                    <Badge
                                                        key={`${keyword}-${index}`}
                                                        className="bg-blue-50 text-blue-600 hover:bg-blue-100"
                                                    >
                                                        {keyword}
                                                    </Badge>
                                                )
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="fulltext">
                            {/* PDF / Full Text View */}
                            <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
                                {/* View Mode Tabs */}
                                {paper.pdfUrl && paper.fullText && (
                                    <div className="flex border-b">
                                        <button
                                            onClick={() => setViewMode("pdf")}
                                            className={`flex-1 py-3 font-medium text-center ${
                                                viewMode === "pdf"
                                                    ? "bg-blue-50 text-blue-600 border-b-2 border-blue-600"
                                                    : "text-gray-600 hover:bg-gray-50"
                                            }`}
                                        >
                                            PDF View
                                        </button>
                                        <button
                                            onClick={() => setViewMode("text")}
                                            className={`flex-1 py-3 font-medium text-center ${
                                                viewMode === "text"
                                                    ? "bg-blue-50 text-blue-600 border-b-2 border-blue-600"
                                                    : "text-gray-600 hover:bg-gray-50"
                                            }`}
                                        >
                                            Text View
                                        </button>
                                    </div>
                                )}

                                {/* Content Based on View Mode */}
                                {viewMode === "pdf" && paper.pdfUrl ? (
                                    <div className="w-full h-screen max-h-[800px]">
                                        <iframe
                                            src={paper.pdfUrl}
                                            className="w-full h-full border-0"
                                            title={paper.title}
                                        />
                                    </div>
                                ) : (
                                    <div className="p-6">
                                        <h2 className="text-2xl font-semibold mb-6">
                                            Full Text
                                        </h2>
                                        <div className="prose max-w-none">
                                            {formatFullText(paper.fullText)}
                                        </div>

                                        {/* Metadata sidebar */}
                                        <div className="mt-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
                                            <h3 className="text-lg font-medium mb-3 text-gray-800">
                                                Paper Details
                                            </h3>
                                            <div className="grid grid-cols-2 gap-y-2 text-sm">
                                                {paper.doi && (
                                                    <>
                                                        <span className="font-medium">
                                                            DOI:
                                                        </span>
                                                        <span>{paper.doi}</span>
                                                    </>
                                                )}
                                                <span className="font-medium">
                                                    Published:
                                                </span>
                                                <span>{paper.date}</span>
                                                <span className="font-medium">
                                                    Journal:
                                                </span>
                                                <span>
                                                    {paper.journal.name}
                                                </span>
                                                <span className="font-medium">
                                                    Citations:
                                                </span>
                                                <span>{paper.citations}</span>
                                            </div>
                                        </div>

                                        {paper.pdfUrl && (
                                            <div className="mt-4 flex justify-center">
                                                <a
                                                    href={paper.pdfUrl}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                >
                                                    <Button className="bg-blue-600 hover:bg-blue-700">
                                                        <Download className="h-4 w-4 mr-2" />
                                                        Download Full PDF
                                                    </Button>
                                                </a>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="metrics">
                            {/* Metrics Section */}
                            <div className="bg-white rounded-lg shadow-sm border p-6">
                                <h2 className="text-xl font-semibold mb-6">
                                    Impact & Metrics
                                </h2>
                                <div className="grid md:grid-cols-3 gap-6">
                                    <div className="p-4 bg-blue-50 rounded-lg">
                                        <div className="text-blue-600 mb-2">
                                            Citations
                                        </div>
                                        <div className="text-2xl font-bold">
                                            {paper.citations}
                                        </div>
                                    </div>
                                    <div className="p-4 bg-green-50 rounded-lg">
                                        <div className="text-green-600 mb-2">
                                            Downloads
                                        </div>
                                        <div className="text-2xl font-bold">
                                            {paper.metrics?.downloads || 0}
                                        </div>
                                    </div>
                                    <div className="p-4 bg-purple-50 rounded-lg">
                                        <div className="text-purple-600 mb-2">
                                            Readers
                                        </div>
                                        <div className="text-2xl font-bold">
                                            {paper.metrics?.readers || 0}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="comments">
                            {/* Comments Section */}
                            <div className="bg-white rounded-lg shadow-sm border p-6">
                                <div className="flex justify-between items-center mb-6">
                                    <h2 className="text-xl font-semibold">
                                        Comments & Discussion
                                    </h2>
                                    <Button
                                        onClick={() => setShowCommentForm(true)}
                                    >
                                        <MessageSquare className="h-4 w-4 mr-2" />
                                        Add Comment
                                    </Button>
                                </div>

                                <div className="space-y-6">
                                    {paper.comments?.map((comment) => (
                                        <div
                                            key={comment.id}
                                            className="border-b pb-6 last:border-0"
                                        >
                                            <div className="flex items-start space-x-4">
                                                <div className="h-10 w-10 rounded-full bg-gray-100"></div>
                                                <div className="flex-1">
                                                    <div className="flex items-center justify-between">
                                                        <h3 className="font-medium">
                                                            {comment.author}
                                                        </h3>
                                                        <span className="text-sm text-gray-500">
                                                            {comment.date}
                                                        </span>
                                                    </div>
                                                    <p className="mt-2 text-gray-600">
                                                        {comment.content}
                                                    </p>
                                                    <div className="mt-3 flex items-center gap-4">
                                                        <button className="text-sm text-gray-500 flex items-center gap-1">
                                                            <ThumbsUp className="h-4 w-4" />
                                                            {comment.likes}
                                                        </button>
                                                        <button className="text-sm text-gray-500">
                                                            Reply
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Add comment form */}
                                {showCommentForm && (
                                    <div className="mt-8 p-4 border rounded-lg">
                                        <h3 className="font-medium mb-3">
                                            Add Your Comment
                                        </h3>
                                        <textarea
                                            className="w-full border rounded-md p-3 mb-3 h-32"
                                            placeholder="Share your thoughts on this paper..."
                                        ></textarea>
                                        <div className="flex justify-end gap-3">
                                            <Button
                                                variant="outline"
                                                onClick={() =>
                                                    setShowCommentForm(false)
                                                }
                                            >
                                                Cancel
                                            </Button>
                                            <Button className="bg-blue-600 hover:bg-blue-700">
                                                Submit Comment
                                            </Button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </TabsContent>
                    </Tabs>

                    {/* Related Papers */}
                    <div className="bg-white rounded-lg shadow-sm border p-6 mt-8">
                        <h2 className="text-xl font-semibold mb-6">
                            Related Papers
                        </h2>
                        {relatedPapers.length > 0 ? (
                            <div className="grid md:grid-cols-2 gap-6">
                                {relatedPapers.map((related) => (
                                    <Link
                                        key={related.id}
                                        to={`/research-papers/${related.id}`}
                                    >
                                        <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
                                            <h3 className="font-medium mb-2">
                                                {related.title}
                                            </h3>
                                            <p className="text-sm text-gray-600 mb-3">
                                                {related.authors.join(", ")}
                                            </p>
                                            <div className="flex items-center gap-2 text-sm text-gray-500">
                                                <Calendar className="h-4 w-4" />
                                                {related.date}
                                            </div>
                                        </div>
                                    </Link>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center text-gray-500 py-4">
                                <p>No related papers found</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
