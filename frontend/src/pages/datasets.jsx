import React, { useState, useEffect, Fragment } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import {
    Search,
    Upload,
    FileType,
    Calendar,
    ChevronLeft,
    ChevronRight,
    BookOpen,
    Database,
    User,
    Filter,
    X,
} from "lucide-react";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const currentUser = {
    id: "user123",
    name: "Jane Researcher",
    avatar: "https://i.pravatar.cc/300",
};

const Datasets = () => {
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedTags, setSelectedTags] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [currentPage, setCurrentPage] = useState(1);
    const [activeTab, setActiveTab] = useState("all");
    const [datasets, setDatasets] = useState([]);
    const [userDatasets, setUserDatasets] = useState([]);
    const [allTags, setAllTags] = useState([]);
    const [totalResults, setTotalResults] = useState(0);

    const itemsPerPage = 20;

    // Function to fetch data from OpenAlex API
    const fetchOpenAlexData = async (page = 1, query = "") => {
        setIsLoading(true);
        try {
            // Construct the OpenAlex API URL with pagination and filters
            let url = `https://api.openalex.org/works?per_page=${itemsPerPage}&page=${page}&search=dataset`;

            // Add search query if provided
            if (query) {
                url += `&search=${encodeURIComponent(query)}`;
            }

            // Add selected concepts/tags as filters if any
            if (selectedTags.length > 0) {
                const conceptsQuery = selectedTags
                    .map(
                        (tag) =>
                            `concept.display_name:${encodeURIComponent(tag)}`
                    )
                    .join(" OR ");
                url += `&filter=${encodeURIComponent(conceptsQuery)}`;
            }

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`OpenAlex API error: ${response.status}`);
            }

            const data = await response.json();

            // Transform OpenAlex data to match our component's expected format
            const formattedData = data.results.map((work) => ({
                id: `oalex-${work.id.split("/").pop()}`, // Extract just the ID part
                title: work.title || "Untitled Research",
                description: work.abstract || "No abstract available.",
                tags: extractConcepts(work),
                files: extractFiles(work),
                created_at: work.publication_date || new Date().toISOString(),
                source: "OpenAlex",
                citations: work.cited_by_count || 0,
                author: extractAuthors(work),
                institution: extractInstitutions(work),
                doi: work.doi,
                url:
                    work.open_access?.oa_url ||
                    work.primary_location?.landing_page_url,
            }));

            setDatasets(formattedData);
            setTotalResults(data.meta?.count || formattedData.length);

            // Extract all unique concepts/topics to use as tags
            const tags = new Set(
                formattedData.flatMap((dataset) => dataset.tags)
            );
            if (tags.size > 0) {
                setAllTags([...tags]);
            }

            setIsLoading(false);
        } catch (error) {
            console.error("Error fetching OpenAlex data:", error);
            setIsLoading(false);
        }
    };

    // Helper to extract concepts as tags
    const extractConcepts = (work) => {
        if (!work.concepts || work.concepts.length === 0) {
            return ["research"];
        }

        // Extract top concepts (limit to prevent too many tags)
        return work.concepts
            .slice(0, 3)
            .map((concept) => concept.display_name)
            .filter(Boolean);
    };

    // Helper to extract files information
    const extractFiles = (work) => {
        const files = [];

        // Add primary PDF if available
        if (work.open_access?.oa_url) {
            files.push({
                id: `file-pdf-${work.id}`,
                name: "Full Text PDF",
                type: "PDF",
                url: work.open_access.oa_url,
            });
        }

        // Add datasets if available
        if (work.referenced_works && work.referenced_works.length > 0) {
            files.push({
                id: `file-ref-${work.id}`,
                name: "Referenced Works",
                type: "REF",
                count: work.referenced_works.length,
            });
        }

        // If no files found, add a placeholder
        if (files.length === 0) {
            files.push({
                id: `file-meta-${work.id}`,
                name: "Metadata Only",
                type: "META",
            });
        }

        return files;
    };

    // Helper to extract authors information
    const extractAuthors = (work) => {
        if (!work.authorships || work.authorships.length === 0) {
            return "Unknown Author";
        }

        // Get first author
        const firstAuthor = work.authorships[0];

        if (work.authorships.length === 1) {
            return firstAuthor.author?.display_name || "Unknown Author";
        }

        // First author + et al. if multiple authors
        return `${firstAuthor.author?.display_name || "Unknown"} et al.`;
    };

    // Helper to extract institutions
    const extractInstitutions = (work) => {
        if (!work.authorships || work.authorships.length === 0) {
            return "";
        }

        // Try to get institution from first author
        const firstAuthor = work.authorships[0];
        if (firstAuthor.institutions && firstAuthor.institutions.length > 0) {
            return firstAuthor.institutions[0].display_name || "";
        }

        return "";
    };

    // Fetch user datasets (simulated - would connect to your backend in real implementation)
    const fetchUserDatasets = async () => {
        try {
            // Simulate API call with timeout
            await new Promise((resolve) => setTimeout(resolve, 500));

            // Generate sample user datasets (in a real implementation, this would be an API call to your backend)
            const userDatasetsSample = Array.from({ length: 8 }, (_, i) => ({
                id: `user-${i + 1}`,
                title: `My Research Project ${i + 1}`,
                description: `Personal dataset for my research on ${
                    [
                        "neural networks",
                        "human behavior",
                        "molecular structures",
                        "environmental impacts",
                        "economic patterns",
                    ][i % 5]
                }.`,
                tags: [
                    ["personal", "private", "collaborative"][i % 3],
                    ["experiment", "analysis", "survey"][i % 3],
                    ["draft", "complete", "in-progress"][i % 3],
                ],
                files: Array.from(
                    { length: Math.floor(Math.random() * 5) + 1 },
                    (_, j) => ({
                        id: `user-file-${i}-${j}`,
                        name: `my-data-${i}-${j}.${
                            ["csv", "json", "xlsx", "txt"][j % 4]
                        }`,
                        size: Math.floor(Math.random() * 500000),
                        type: ["CSV", "JSON", "XLSX", "TXT"][j % 4],
                    })
                ),
                created_at: new Date(
                    2024,
                    Math.floor(Math.random() * 3),
                    Math.floor(Math.random() * 28) + 1
                ).toISOString(),
                source: "User Upload",
                author: currentUser.name,
                isUserOwned: true,
            }));

            setUserDatasets(userDatasetsSample);

            // Add user tags to all tags
            const userTags = new Set(
                userDatasetsSample.flatMap((dataset) => dataset.tags)
            );
            setAllTags((prevTags) => [...new Set([...prevTags, ...userTags])]);
        } catch (error) {
            console.error("Error fetching user data:", error);
        }
    };

    // Initial data fetch
    useEffect(() => {
        fetchOpenAlexData();
        fetchUserDatasets();
    }, []);

    // Refetch when search, tags, or pagination changes
    useEffect(() => {
        if (activeTab !== "user") {
            fetchOpenAlexData(currentPage, searchQuery);
        }
    }, [currentPage, searchQuery, selectedTags, activeTab]);

    // Filter datasets based on search query and selected tags
    const getFilteredDatasets = () => {
        // If on OpenAlex tab, we're already filtering via API
        if (activeTab === "openalex") {
            return datasets;
        }

        // If on User tab, we need to filter locally
        if (activeTab === "user") {
            return userDatasets.filter((dataset) => {
                const matchesSearch =
                    !searchQuery ||
                    dataset.title
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase()) ||
                    dataset.description
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase());
                const matchesTags =
                    selectedTags.length === 0 ||
                    selectedTags.every((tag) => dataset.tags.includes(tag));
                return matchesSearch && matchesTags;
            });
        }

        // If on All tab, combine both sources (OpenAlex already filtered via API)
        const filteredUserDatasets = userDatasets.filter((dataset) => {
            const matchesSearch =
                !searchQuery ||
                dataset.title
                    .toLowerCase()
                    .includes(searchQuery.toLowerCase()) ||
                dataset.description
                    .toLowerCase()
                    .includes(searchQuery.toLowerCase());
            const matchesTags =
                selectedTags.length === 0 ||
                selectedTags.every((tag) => dataset.tags.includes(tag));
            return matchesSearch && matchesTags;
        });

        return [...datasets, ...filteredUserDatasets];
    };

    const filteredDatasets = getFilteredDatasets();

    // Pagination logic
    const totalPages = Math.ceil(filteredDatasets.length / itemsPerPage);
    const currentDatasets = filteredDatasets.slice(0, itemsPerPage);

    const toggleTag = (tag) => {
        setSelectedTags((prev) =>
            prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
        );
        setCurrentPage(1); // Reset to first page when filter changes
    };

    const clearFilters = () => {
        setSelectedTags([]);
        setSearchQuery("");
        setCurrentPage(1);
    };

    // Card animation variants
    const cardVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: (i) => ({
            opacity: 1,
            y: 0,
            transition: {
                delay: i * 0.05,
                duration: 0.3,
            },
        }),
        exit: { opacity: 0, y: -20, transition: { duration: 0.2 } },
    };

    const getCardLink = (dataset) => {
        // Always show detailed view first
        return `/dataset/${dataset.id}`;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 py-8 px-4 sm:px-6 lg:px-8">
            <div className="max-w-7xl mx-auto">
                {/* Header with Upload Button */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
                    <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl md:text-5xl mb-4 md:mb-0">
                        Research Datasets
                    </h1>
                    <Link to="/dataset-upload">
                        <Button className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md shadow-sm transition-all duration-200 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                            <Upload className="h-4 w-4 mr-2 inline-block align-middle" />
                            Upload Dataset
                        </Button>
                    </Link>
                </div>

                {/* Search and Filter */}
                <Card className="mb-6 border-none shadow-md">
                    <CardContent className="p-6">
                        <div className="mb-6">
                            <label htmlFor="search" className="sr-only">
                                Search
                            </label>
                            <div className="relative rounded-md shadow-sm">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Search className="h-5 w-5 text-gray-400" />
                                </div>
                                <Input
                                    id="search"
                                    type="text"
                                    placeholder="Search by title, description, or author..."
                                    value={searchQuery}
                                    onChange={(e) => {
                                        setSearchQuery(e.target.value);
                                        setCurrentPage(1);
                                    }}
                                    className="pl-10 pr-4 py-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 w-full"
                                />
                            </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-3">
                            <div className="flex items-center mr-2">
                                <Filter className="h-4 w-4 text-gray-500 mr-1" />
                                <span className="text-sm font-medium text-gray-600">
                                    Filter by concepts:
                                </span>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {allTags.slice(0, 15).map((tag) => (
                                    <Badge
                                        key={tag}
                                        variant={
                                            selectedTags.includes(tag)
                                                ? "default"
                                                : "outline"
                                        }
                                        className={`${
                                            selectedTags.includes(tag)
                                                ? "bg-blue-600 text-white hover:bg-blue-700"
                                                : "hover:bg-blue-50"
                                        } text-sm font-medium px-3 py-1 rounded-full cursor-pointer transition-colors duration-200`}
                                        onClick={() => toggleTag(tag)}
                                    >
                                        {tag}
                                    </Badge>
                                ))}
                                {allTags.length > 15 && (
                                    <DropdownMenu>
                                        <DropdownMenuTrigger asChild>
                                            <Badge
                                                variant="outline"
                                                className="cursor-pointer hover:bg-blue-50"
                                            >
                                                +{allTags.length - 15} more
                                                concepts
                                            </Badge>
                                        </DropdownMenuTrigger>
                                        <DropdownMenuContent
                                            align="end"
                                            className="w-56 p-2 max-h-64 overflow-y-auto"
                                        >
                                            {allTags.slice(15).map((tag) => (
                                                <DropdownMenuItem
                                                    key={tag}
                                                    className={
                                                        selectedTags.includes(
                                                            tag
                                                        )
                                                            ? "bg-blue-50"
                                                            : ""
                                                    }
                                                    onClick={() =>
                                                        toggleTag(tag)
                                                    }
                                                >
                                                    {tag}
                                                </DropdownMenuItem>
                                            ))}
                                        </DropdownMenuContent>
                                    </DropdownMenu>
                                )}
                            </div>

                            {(selectedTags.length > 0 || searchQuery) && (
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={clearFilters}
                                    className="ml-2 text-sm"
                                >
                                    <X className="h-3 w-3 mr-1" />
                                    Clear filters
                                </Button>
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* Modified Tabs */}
                <Tabs
                    defaultValue="all"
                    value={activeTab}
                    onValueChange={(value) => {
                        setActiveTab(value);
                        setCurrentPage(1);
                    }}
                    className="mb-6"
                >
                    <TabsList className="grid w-full md:w-auto grid-cols-2 rounded-lg bg-gray-100 p-1">
                        <TabsTrigger
                            value="all"
                            className="rounded-md data-[state=active]:bg-white data-[state=active]:shadow-sm"
                        >
                            <Database className="h-4 w-4 mr-1.5" />
                            OpenAlex
                        </TabsTrigger>
                        <TabsTrigger
                            value="user"
                            className="rounded-md data-[state=active]:bg-white data-[state=active]:shadow-sm"
                        >
                            <User className="h-4 w-4 mr-1.5" />
                            Your Uploads
                        </TabsTrigger>
                    </TabsList>

                    <TabsContent value="all" className="mt-6">
                        <div className="flex justify-between items-center mb-4">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                                    <Database className="h-5 w-5 mr-2 text-blue-600" />
                                    OpenAlex Datasets
                                </h2>
                                <p className="text-sm text-gray-500 mt-1">
                                    Data from OpenAlex, a free and open index of
                                    scholarly research
                                </p>
                            </div>
                            <span className="text-sm text-gray-500">
                                Showing {filteredDatasets.length} of{" "}
                                {totalResults} results
                            </span>
                        </div>
                    </TabsContent>

                    <TabsContent value="user" className="mt-6">
                        <div className="flex justify-between items-center mb-4">
                            <div>
                                <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                                    <User className="h-5 w-5 mr-2 text-blue-600" />
                                    Your Uploaded Datasets
                                </h2>
                                <p className="text-sm text-gray-500 mt-1">
                                    Datasets you've personally uploaded to the
                                    platform
                                </p>
                            </div>
                            <span className="text-sm text-gray-500">
                                Showing {filteredDatasets.length} results
                            </span>
                        </div>
                    </TabsContent>
                </Tabs>

                {/* Loading State */}
                {isLoading ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {Array.from({ length: 6 }).map((_, index) => (
                            <Card key={index} className="overflow-hidden">
                                <CardHeader className="pb-2">
                                    <Skeleton className="h-6 w-3/4 mb-2" />
                                    <Skeleton className="h-4 w-1/2" />
                                </CardHeader>
                                <CardContent>
                                    <Skeleton className="h-4 w-full mb-2" />
                                    <Skeleton className="h-4 w-full mb-2" />
                                    <Skeleton className="h-4 w-3/4 mb-4" />
                                    <div className="flex gap-2 mb-4">
                                        <Skeleton className="h-6 w-16 rounded-full" />
                                        <Skeleton className="h-6 w-16 rounded-full" />
                                        <Skeleton className="h-6 w-16 rounded-full" />
                                    </div>
                                    <div className="flex justify-between">
                                        <Skeleton className="h-4 w-1/4" />
                                        <Skeleton className="h-4 w-1/4" />
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                ) : (
                    <>
                        {/* Dataset Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                            <AnimatePresence>
                                {currentDatasets.length > 0 ? (
                                    currentDatasets.map((dataset, index) => (
                                        <motion.div
                                            key={dataset.id}
                                            custom={index}
                                            variants={cardVariants}
                                            initial="hidden"
                                            animate="visible"
                                            exit="exit"
                                            layout
                                        >
                                            <Link
                                                to={getCardLink(dataset)}
                                                className="block h-full"
                                                target={
                                                    !dataset.isUserOwned
                                                        ? "_blank"
                                                        : "_self"
                                                }
                                            >
                                                <Card className="h-full overflow-hidden hover:shadow-lg transition-shadow duration-300 border-gray-200 bg-white">
                                                    <CardHeader
                                                        className={`pb-2 ${
                                                            dataset.isUserOwned
                                                                ? "bg-blue-50"
                                                                : ""
                                                        }`}
                                                    >
                                                        <div className="flex justify-between items-start">
                                                            <CardTitle className="text-xl font-semibold text-gray-900 line-clamp-2">
                                                                {dataset.title}
                                                            </CardTitle>
                                                            {dataset.isUserOwned ? (
                                                                <Badge
                                                                    variant="outline"
                                                                    className="bg-blue-100 text-blue-800 border-blue-200"
                                                                >
                                                                    Your Upload
                                                                </Badge>
                                                            ) : (
                                                                <Badge
                                                                    variant="outline"
                                                                    className="bg-green-100 text-green-800 border-green-200"
                                                                >
                                                                    OpenAlex
                                                                </Badge>
                                                            )}
                                                        </div>
                                                        <CardDescription className="text-sm text-gray-500 flex items-center gap-1">
                                                            <Database className="h-3 w-3" />
                                                            {dataset.source ||
                                                                "Research Dataset"}
                                                            {dataset.citations >
                                                                0 &&
                                                                ` • ${dataset.citations} citations`}
                                                        </CardDescription>
                                                    </CardHeader>
                                                    <CardContent className="pt-4">
                                                        <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                                                            {
                                                                dataset.description
                                                            }
                                                        </p>

                                                        {/* Author & Institution */}
                                                        {dataset.author && (
                                                            <p className="text-gray-700 text-sm mb-3 font-medium">
                                                                By:{" "}
                                                                {dataset.author}
                                                                {dataset.institution &&
                                                                    ` • ${dataset.institution}`}
                                                            </p>
                                                        )}

                                                        {/* Tags */}
                                                        <div className="flex flex-wrap gap-1.5 mb-4">
                                                            {dataset.tags.map(
                                                                (
                                                                    tag,
                                                                    index
                                                                ) => (
                                                                    <Badge
                                                                        key={
                                                                            index
                                                                        }
                                                                        variant="secondary"
                                                                        className="bg-gray-100 text-gray-600 text-xs font-medium px-2 py-0.5 rounded-full"
                                                                    >
                                                                        {tag}
                                                                    </Badge>
                                                                )
                                                            )}
                                                        </div>

                                                        {/* File Info */}
                                                        <div className="flex items-center justify-between text-xs text-gray-500">
                                                            <div className="flex items-center">
                                                                <FileType className="h-3.5 w-3.5 mr-1" />
                                                                <span>
                                                                    {
                                                                        dataset
                                                                            .files
                                                                            .length
                                                                    }{" "}
                                                                    {dataset
                                                                        .files
                                                                        .length ===
                                                                    1
                                                                        ? "file"
                                                                        : "files"}
                                                                </span>
                                                                {dataset.doi && (
                                                                    <span className="ml-2">
                                                                        • DOI:{" "}
                                                                        {
                                                                            dataset.doi
                                                                        }
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <div className="flex items-center">
                                                                <Calendar className="h-3.5 w-3.5 mr-1" />
                                                                <span>
                                                                    {new Date(
                                                                        dataset.created_at
                                                                    ).toLocaleDateString()}
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </CardContent>
                                                </Card>
                                            </Link>
                                        </motion.div>
                                    ))
                                ) : (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="col-span-1 md:col-span-2 lg:col-span-3 text-center py-16"
                                    >
                                        <div className="bg-white rounded-lg shadow-sm p-8">
                                            <p className="text-gray-600 mb-4">
                                                No datasets found matching your
                                                criteria
                                            </p>
                                            <Button
                                                variant="outline"
                                                onClick={clearFilters}
                                                className="text-sm"
                                            >
                                                Clear filters
                                            </Button>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div className="flex justify-center items-center mt-8 space-x-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() =>
                                        setCurrentPage((prev) =>
                                            Math.max(prev - 1, 1)
                                        )
                                    }
                                    disabled={currentPage === 1}
                                    className="px-2.5"
                                >
                                    <ChevronLeft className="h-4 w-4" />
                                </Button>
                                <div className="flex items-center space-x-1">
                                    {Array.from(
                                        { length: totalPages },
                                        (_, i) => i + 1
                                    )
                                        .filter((page) => {
                                            // Show first page, last page, current page, and siblings
                                            return (
                                                page === 1 ||
                                                page === totalPages ||
                                                Math.abs(page - currentPage) <=
                                                    1
                                            );
                                        })
                                        .map((page, index, array) => {
                                            // Add ellipsis where needed
                                            const prevPage = array[index - 1];
                                            return (
                                                <Fragment key={page}>
                                                    {prevPage &&
                                                        page - prevPage > 1 && (
                                                            <span className="px-3 py-2 text-sm text-gray-500">
                                                                …
                                                            </span>
                                                        )}
                                                    <Button
                                                        variant={
                                                            currentPage === page
                                                                ? "default"
                                                                : "outline"
                                                        }
                                                        size="sm"
                                                        onClick={() =>
                                                            setCurrentPage(page)
                                                        }
                                                        className={`min-w-8 h-8 px-3 text-sm ${
                                                            currentPage === page
                                                                ? "bg-blue-600 text-white"
                                                                : "text-gray-600"
                                                        }`}
                                                    >
                                                        {page}
                                                    </Button>
                                                </Fragment>
                                            );
                                        })}
                                </div>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() =>
                                        setCurrentPage((prev) =>
                                            Math.min(prev + 1, totalPages)
                                        )
                                    }
                                    disabled={currentPage === totalPages}
                                    className="px-2.5"
                                >
                                    <ChevronRight className="h-4 w-4" />
                                </Button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default Datasets;
