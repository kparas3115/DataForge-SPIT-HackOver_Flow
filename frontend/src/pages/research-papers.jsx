import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
    Search,
    BookOpen,
    Calendar,
    ArrowRight,
    SortAsc,
    SortDesc,
    Filter,
    Loader2,
} from "lucide-react";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Link } from "react-router-dom";
import _ from "lodash"; // Make sure lodash is installed

// CORE API key
const API_KEY = "JkqyH1pgKdM8vFXAB6LxeZEozGQfcPOr";
const CORE_API_URL = "https://api.core.ac.uk/v3";

export default function ResearchPapers() {
    const [scrolled, setScrolled] = useState(false);
    const [papers, setPapers] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [debouncedSearchQuery, setDebouncedSearchQuery] = useState("");
    const [sortConfig, setSortConfig] = useState({
        field: "publishedDate",
        direction: "desc",
    });
    const [filterType, setFilterType] = useState("all");
    const [filterField, setFilterField] = useState("all");
    const [loading, setLoading] = useState(false);
    const [page, setPage] = useState(1);
    const [totalResults, setTotalResults] = useState(0);
    const [shouldFetch, setShouldFetch] = useState(true);
    const resultsPerPage = 18;

    const sortOptions = [
        { label: "Date", value: "publishedDate" },
        { label: "Title", value: "title" },
        { label: "Citations", value: "citationCount" },
        { label: "Downloads", value: "downloadCount" },
    ];

    const filterTypes = [
        { label: "All Types", value: "all" },
        { label: "Journal Article", value: "journal-article" },
        { label: "Conference Paper", value: "conference-paper" },
        { label: "Book Chapter", value: "book-chapter" },
        { label: "Thesis", value: "thesis" },
    ];

    const techFields = [
        { label: "All Fields", value: "all" },
        { label: "Quantum Computing", value: "Quantum Computing" },
        { label: "Machine Learning", value: "Machine Learning" },
        { label: "Artificial Intelligence", value: "Artificial Intelligence" },
        { label: "Deep Learning", value: "Deep Learning" },
        { label: "Generative AI", value: "Generative AI" },
        { label: "Cloud Computing", value: "Cloud Computing" },
        { label: "Physics", value: "Physics" },
        { label: "Neural Networks", value: "Neural Networks" },
        { label: "Natural Language Processing", value: "NLP" },
        { label: "Computer Vision", value: "Computer Vision" },
        { label: "Robotics", value: "Robotics" },
    ];

    // Debounce search input
    useEffect(() => {
        const debounceTimer = setTimeout(() => {
            setDebouncedSearchQuery(searchQuery);
        }, 500);

        return () => {
            clearTimeout(debounceTimer);
        };
    }, [searchQuery]);

    // Memoized fetch function to prevent recreation on every render
    const fetchPapers = useCallback(async () => {
        if (!shouldFetch) return;

        setLoading(true);
        setShouldFetch(false);

        try {
            // Construct the search query
            let searchQuery = debouncedSearchQuery.trim()
                ? debouncedSearchQuery
                : "*";

            // Add field filter if selected
            if (filterField !== "all") {
                searchQuery = `${searchQuery} AND (${filterField})`;
            }

            // Add type filter if selected
            const typeParam = filterType !== "all" ? `&type=${filterType}` : "";

            // Add language filter for English only
            const languageParam = "&language=en";

            // Add additional query parameters to filter out invalid titles
            searchQuery = `${searchQuery} AND NOT title:"*"`;

            // Calculate offset for pagination
            const offset = (page - 1) * resultsPerPage;

            // Make the API request
            const requestUrl = `${CORE_API_URL}/search/works?q=${encodeURIComponent(
                searchQuery
            )}${typeParam}${languageParam}&offset=${offset}&limit=${resultsPerPage}&sort=${
                sortConfig.field
            }:${sortConfig.direction}`;

            console.log("Fetching from:", requestUrl);

            const response = await fetch(requestUrl, {
                headers: {
                    Authorization: `Bearer ${API_KEY}`,
                },
            });

            const data = await response.json();

            if (!data.results || !Array.isArray(data.results)) {
                console.error("Invalid response format:", data);
                alert("Error: Invalid response format from CORE API");
                return;
            }

            // Process the results and filter out papers with invalid titles
            const processedPapers = data.results
                .filter((paper) => {
                    return (
                        paper.title &&
                        paper.title !== "*" &&
                        paper.title.length > 2 &&
                        !paper.title.includes("???")
                    );
                })
                .map((paper) => {
                    // Extract author names
                    const authors =
                        paper.authors?.map((author) => author.name) || [];

                    // Format date
                    const date = paper.publishedDate
                        ? new Date(paper.publishedDate).toLocaleDateString(
                              "en-US",
                              {
                                  year: "numeric",
                                  month: "short",
                                  day: "numeric",
                              }
                          )
                        : "Unknown Date";

                    // Extract just a few important tags
                    const tags = [
                        ...(paper.subjects || []),
                        ...(paper.topics || []),
                    ]
                        .filter(
                            (tag) => tag && tag.length > 0 && tag.length < 30
                        )
                        .slice(0, 3);

                    return {
                        id: paper.id,
                        title: paper.title || "Untitled Paper",
                        abstract: paper.abstract || "No abstract available",
                        authors: authors,
                        date: date,
                        field: paper.fieldsOfStudy?.[0] || "General",
                        tags: tags,
                        citationCount: paper.citationCount || 0,
                    };
                });

            setPapers(processedPapers);
            setTotalResults(data.totalHits || 0);
        } catch (error) {
            console.error("Error fetching papers from CORE API:", error);
            alert(`Error fetching papers: ${error.message}`);
            setPapers([]);
        } finally {
            setLoading(false);
        }
    }, [
        debouncedSearchQuery,
        filterField,
        filterType,
        sortConfig,
        page,
        shouldFetch,
    ]);

    // Only trigger fetch when search parameters change
    useEffect(() => {
        setShouldFetch(true);
    }, [debouncedSearchQuery, filterField, filterType, sortConfig, page]);

    // Fetch data when shouldFetch changes to true
    useEffect(() => {
        if (shouldFetch) {
            fetchPapers();
        }
    }, [shouldFetch, fetchPapers]);

    // Handle sort change
    const handleSort = (field) => {
        setSortConfig({
            field,
            direction:
                sortConfig.field === field && sortConfig.direction === "asc"
                    ? "desc"
                    : "asc",
        });
    };

    // Handle search submission
    const handleSearch = (e) => {
        e.preventDefault();
        setPage(1); // Reset to first page when searching
        setShouldFetch(true);
    };

    // Handle scroll event for header styling
    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 pt-28 pb-16">
                {/* Hero Section */}
                <div className="max-w-3xl mx-auto text-center mb-16">
                    <h1 className="text-4xl font-bold text-slate-800 mb-4 leading-tight">
                        Research Repository
                    </h1>
                    <p className="text-lg text-slate-600 leading-relaxed">
                        Powered by CORE API - Access to Millions of Open Access
                        Research Papers
                    </p>
                </div>

                {/* Search and Filter Section */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6 mb-12">
                    <form onSubmit={handleSearch}>
                        <div className="flex flex-col md:flex-row gap-6">
                            <div className="flex-1 relative">
                                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-5 w-5" />
                                <Input
                                    placeholder="Search papers, authors, or keywords..."
                                    className="pl-10 h-12 border-slate-200 rounded-lg text-slate-600 placeholder:text-slate-400"
                                    value={searchQuery}
                                    onChange={(e) =>
                                        setSearchQuery(e.target.value)
                                    }
                                />
                            </div>
                            <div className="flex gap-4">
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button
                                            variant="outline"
                                            className="gap-2"
                                        >
                                            <Filter className="h-4 w-4" />
                                            {filterField === "all"
                                                ? "All Fields"
                                                : filterField}
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent
                                        align="end"
                                        className="w-56"
                                    >
                                        <DropdownMenuLabel>
                                            Filter by Field
                                        </DropdownMenuLabel>
                                        <DropdownMenuSeparator />
                                        {techFields.map((field) => (
                                            <DropdownMenuItem
                                                key={field.value}
                                                onClick={() => {
                                                    setFilterField(field.value);
                                                    setPage(1);
                                                }}
                                            >
                                                {field.label}
                                            </DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>

                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button
                                            variant="outline"
                                            className="gap-2"
                                        >
                                            {sortConfig.direction === "asc" ? (
                                                <SortAsc className="h-4 w-4" />
                                            ) : (
                                                <SortDesc className="h-4 w-4" />
                                            )}
                                            Sort by{" "}
                                            {
                                                sortOptions.find(
                                                    (opt) =>
                                                        opt.value ===
                                                        sortConfig.field
                                                )?.label
                                            }
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent
                                        align="end"
                                        className="w-56"
                                    >
                                        <DropdownMenuLabel>
                                            Sort by
                                        </DropdownMenuLabel>
                                        <DropdownMenuSeparator />
                                        {sortOptions.map((option) => (
                                            <DropdownMenuItem
                                                key={option.value}
                                                onClick={() => {
                                                    handleSort(option.value);
                                                    setPage(1);
                                                }}
                                            >
                                                {option.label}
                                            </DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>

                                <Button
                                    type="submit"
                                    className="bg-blue-600 hover:bg-blue-700 text-white"
                                    disabled={loading}
                                >
                                    {loading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Searching
                                        </>
                                    ) : (
                                        "Search"
                                    )}
                                </Button>
                            </div>
                        </div>
                    </form>
                </div>

                {/* Results Summary */}
                <div className="flex justify-between items-center mb-8">
                    <p className="text-slate-600">
                        {loading
                            ? "Searching..."
                            : `Showing ${papers.length} of ${totalResults} results`}
                    </p>
                </div>

                {/* Loading Indicator */}
                {loading && (
                    <div className="flex justify-center items-center py-12">
                        <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
                        <span className="ml-2 text-slate-600">
                            Loading research papers...
                        </span>
                    </div>
                )}

                {/* No Results Message */}
                {!loading && papers.length === 0 && (
                    <div className="text-center py-16 bg-white rounded-xl shadow-sm border border-slate-100">
                        <BookOpen className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                        <h3 className="text-xl font-medium text-slate-800 mb-2">
                            No papers found
                        </h3>
                        <p className="text-slate-600 max-w-md mx-auto">
                            Try adjusting your search or filter criteria to find
                            more research papers.
                        </p>
                    </div>
                )}

                {/* Papers Grid */}
                {!loading && papers.length > 0 && (
                    <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                        {papers.map((paper, index) => (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{
                                    duration: 0.3,
                                    delay: index * 0.05,
                                }}
                                key={paper.id || index}
                            >
                                <Card className="group h-full bg-white border-slate-100 hover:shadow-lg transition-all duration-300">
                                    <CardHeader className="space-y-3 pb-4">
                                        <motion.div
                                            whileHover={{ scale: 1.01 }}
                                            className="space-y-3"
                                        >
                                            <CardTitle className="text-xl font-semibold text-slate-800 leading-tight">
                                                {paper.title}
                                            </CardTitle>
                                            <div className="flex items-center gap-2 text-sm text-slate-500">
                                                <Calendar className="h-4 w-4" />
                                                <span>{paper.date}</span>
                                            </div>
                                        </motion.div>
                                        <div className="flex items-center gap-2 mt-2">
                                            <Badge className="bg-blue-100 text-blue-700 border-none">
                                                {paper.field}
                                            </Badge>
                                            {paper.citationCount > 0 && (
                                                <Badge className="bg-purple-100 text-purple-700 border-none">
                                                    {paper.citationCount}{" "}
                                                    citations
                                                </Badge>
                                            )}
                                        </div>
                                    </CardHeader>
                                    <CardContent className="space-y-6">
                                        {paper.authors &&
                                            paper.authors.length > 0 && (
                                                <div className="space-y-2">
                                                    <div className="flex gap-2 items-center text-slate-600">
                                                        <h4 className="text-sm font-medium">
                                                            Authors
                                                        </h4>
                                                    </div>
                                                    <div className="text-sm text-slate-600 leading-relaxed line-clamp-2">
                                                        {paper.authors.join(
                                                            ", "
                                                        )}
                                                    </div>
                                                </div>
                                            )}

                                        <p className="text-sm text-slate-600 line-clamp-3 leading-relaxed">
                                            {paper.abstract}
                                        </p>

                                        <div className="flex flex-wrap gap-2">
                                            {paper.tags &&
                                                paper.tags.length > 0 &&
                                                paper.tags.map((tag, idx) => (
                                                    <Badge
                                                        key={`${paper.id}-${idx}`}
                                                        className="bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors px-3 py-1 rounded-full text-xs"
                                                    >
                                                        {tag}
                                                    </Badge>
                                                ))}
                                        </div>

                                        <Link
                                            to={`/research-papers/${paper.id}`}
                                            className="w-full"
                                        >
                                            <Button className="w-full bg-slate-50 hover:bg-slate-100 text-slate-600 hover:text-blue-600 border-none group-hover:text-blue-600 transition-colors">
                                                See Details
                                                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                                            </Button>
                                        </Link>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </div>
                )}

                {/* Pagination */}
                {!loading && totalResults > resultsPerPage && (
                    <div className="flex justify-center mt-12">
                        <div className="flex gap-2">
                            <Button
                                variant="outline"
                                onClick={() =>
                                    setPage((prev) => Math.max(prev - 1, 1))
                                }
                                disabled={page === 1 || loading}
                            >
                                Previous
                            </Button>
                            <div className="flex items-center px-4 bg-white border border-slate-200 rounded-md">
                                Page {page} of{" "}
                                {Math.ceil(totalResults / resultsPerPage)}
                            </div>
                            <Button
                                variant="outline"
                                onClick={() =>
                                    setPage((prev) =>
                                        Math.min(
                                            prev + 1,
                                            Math.ceil(
                                                totalResults / resultsPerPage
                                            )
                                        )
                                    )
                                }
                                disabled={
                                    page ===
                                        Math.ceil(
                                            totalResults / resultsPerPage
                                        ) || loading
                                }
                            >
                                Next
                            </Button>
                        </div>
                    </div>
                )}

                {/* Attribution */}
                <div className="text-center mt-16 text-sm text-slate-500">
                    <p>
                        Powered by{" "}
                        <a
                            href="https://core.ac.uk/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline"
                        >
                            CORE
                        </a>{" "}
                        - aggregating the world's open access research papers
                    </p>
                </div>
            </main>
        </div>
    );
}
