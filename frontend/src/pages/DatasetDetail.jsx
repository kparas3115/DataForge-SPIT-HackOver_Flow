import React, { useState, useEffect, Suspense } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Download,
  FileType,
  Calendar,
  ArrowLeft,
  Database,
  Info,
  BookOpen,
} from "lucide-react";
import datasetJson from "../data/datasets.json";

const DatasetDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDataset = async () => {
      setLoading(true);
      try {
        if (!id) {
          navigate("/datasets");
          return;
        }

        // Check if this is an OpenAlex ID
        if (id.startsWith('oalex-')) {
          try {
            const workId = id.replace('oalex-', '');
            // Ensure clean ID format
            const cleanId = workId.replace('https://openalex.org/', '');
            
            const response = await fetch(`https://api.openalex.org/works/${cleanId}`);
            if (!response.ok) throw new Error('Dataset not found');
            const data = await response.json();
            
            // Transform OpenAlex data
            setDataset({
              id: id,
              title: data.title,
              description: data.abstract || "No description available",
              overview: {
                source: "OpenAlex",
                last_updated: new Date(data.updated_date).toLocaleDateString(),
                total_rows: data.referenced_works?.length || 0,
                total_columns: data.concepts?.length || 0,
                data_format: ["PDF", "Metadata"],
                authors: data.authorships?.map(a => a.author.display_name) || [],
                citations: data.cited_by_count || 0,
                published_date: new Date(data.publication_date).toLocaleDateString()
              },
              tags: data.concepts?.map(c => ({
                name: c.display_name,
                score: c.score,
                level: c.level
              })) || [],
              files: [
                ...(data.open_access?.oa_url ? [{
                  name: "Full Text PDF",
                  url: data.open_access.oa_url,
                  type: "PDF"
                }] : []),
                ...(data.primary_location?.pdf_url ? [{
                  name: "Alternative PDF",
                  url: data.primary_location.pdf_url,
                  type: "PDF"
                }] : []),
                ...(data.doi ? [{
                  name: "DOI Link",
                  url: `https://doi.org/${data.doi}`,
                  type: "DOI"
                }] : [])
              ],
              sample_data: {
                columns: data.concepts?.map(c => ({ 
                  name: c.display_name,
                  type: "concept",
                  description: c.level
                })) || [],
                preview: []
              },
              data_dictionary: [
                {
                  column: "Title",
                  type: "text",
                  description: "Research work title",
                  example: data.title
                },
                {
                  column: "Abstract",
                  type: "text",
                  description: "Research abstract",
                  example: data.abstract?.substring(0, 100) + "..."
                },
                ...(data.concepts?.map(c => ({
                  column: c.display_name,
                  type: "concept",
                  description: `Level ${c.level} concept`,
                  example: `Score: ${c.score?.toFixed(2)}`
                })) || [])
              ],
              download_info: {
                download_link: data.open_access?.oa_url || `https://doi.org/${data.doi}`,
                license_type: data.open_access?.license || "Unknown",
                attribution: `Cite as: ${data.authorships?.[0]?.author?.display_name || "Authors"} (${new Date(data.publication_date).getFullYear()})`
              },
              use_cases: [
                "Research Reference",
                "Literature Review",
                "Citation Analysis",
                "Bibliometric Studies"
              ],
              user_feedback: {
                comments: [],
                faqs: [
                  {
                    question: "How do I cite this work?",
                    answer: `Use the DOI: ${data.doi}`
                  },
                  {
                    question: "Is this open access?",
                    answer: data.open_access?.is_oa ? "Yes" : "No"
                  }
                ]
              }
            });
          } catch (err) {
            throw new Error(`Failed to fetch OpenAlex dataset: ${err.message}`);
          }
        } else {
          try {
            // Try local storage first
            const localDataset = localStorage.getItem(`dataset-${id}`);
            if (localDataset) {
              setDataset(JSON.parse(localDataset));
            } else {
              const response = await fetch(`/api/datasets/${id}`);
              if (!response.ok) throw new Error('Dataset not found');
              const data = await response.json();
              setDataset(data);
            }
          } catch (err) {
            throw new Error(`Failed to fetch user dataset: ${err.message}`);
          }
        }
      } catch (err) {
        console.error('Error fetching dataset:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDataset();
  }, [id, navigate]);

  const renderExternalLinkButton = (dataset) => {
    if (dataset.url || dataset.download_info?.download_link) {
      return (
        <Button className="ml-4" variant="outline" asChild>
          <a 
            href={dataset.url || dataset.download_info.download_link} 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center"
          >
            <BookOpen className="h-4 w-4 mr-2" />
            View Original Source
          </a>
        </Button>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse space-y-4">
            {/* Add loading skeleton UI here */}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-2xl font-bold text-red-600">Error: {error}</h1>
          <Link to="/datasets">
            <Button variant="ghost" className="mt-4">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Datasets
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  if (!dataset) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header with Back Button and External Link */}
        <div className="flex items-center justify-between mb-6">
          <Link to="/datasets">
            <Button variant="ghost">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Datasets
            </Button>
          </Link>
          {renderExternalLinkButton(dataset)}
        </div>

        {/* Dataset Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">{dataset.title}</h1>
          <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600">
            <div className="flex items-center">
              <Database className="h-4 w-4 mr-1" />
              {dataset.overview.source}
            </div>
            <div className="flex items-center">
              <Calendar className="h-4 w-4 mr-1" />
              Last updated: {dataset.overview.last_updated}
            </div>
            <div className="flex items-center">
              <FileType className="h-4 w-4 mr-1" />
              {dataset.files.length} files
            </div>
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-8">
          {dataset.tags.map((tag) => (
            <Badge key={tag.name} variant="secondary">
              {tag.name}
            </Badge>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Overview Card */}
            <Card>
              <CardHeader>
                <CardTitle>Dataset Overview</CardTitle>
                <CardDescription>{dataset.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Total Rows</p>
                    <p className="text-lg font-semibold">{dataset.overview.total_rows.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Total Columns</p>
                    <p className="text-lg font-semibold">{dataset.overview.total_columns}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Data Formats</p>
                    <p className="text-lg font-semibold">{dataset.overview.data_format.join(", ")}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Sample Data Card */}
            <Card>
              <CardHeader>
                <CardTitle>Sample Data Preview</CardTitle>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {dataset.sample_data.columns.map((col) => (
                        <TableHead key={col.name}>{col.name}</TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dataset.sample_data.preview.map((row, idx) => (
                      <TableRow key={idx}>
                        {Object.values(row).map((value, i) => (
                          <TableCell key={i}>{value}</TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Data Dictionary */}
            <Card>
              <CardHeader>
                <CardTitle>Data Dictionary</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Column</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Example</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dataset.data_dictionary.map((item) => (
                      <TableRow key={item.column}>
                        <TableCell className="font-medium">{item.column}</TableCell>
                        <TableCell>{item.type}</TableCell>
                        <TableCell>{item.description}</TableCell>
                        <TableCell>{item.example}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
            {/* Download Card */}
            <Card>
              <CardHeader>
                <CardTitle>Access Dataset</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dataset.download_info?.download_link && (
                    <Button className="w-full" asChild>
                      <a 
                        href={dataset.download_info.download_link} 
                        download 
                        className="flex items-center justify-center"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download Files
                      </a>
                    </Button>
                  )}
                  {dataset.url && (
                    <Button className="w-full" variant="outline" asChild>
                      <a 
                        href={dataset.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center justify-center"
                      >
                        <BookOpen className="h-4 w-4 mr-2" />
                        View on Original Platform
                      </a>
                    </Button>
                  )}
                  <div className="text-sm text-gray-500">
                    <p className="mb-2">
                      <span className="font-medium">License:</span> {dataset.download_info?.license_type || "Not specified"}
                    </p>
                    <p>
                      <span className="font-medium">Attribution:</span> {dataset.download_info?.attribution || "Not specified"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Use Cases Card */}
            <Card>
              <CardHeader>
                <CardTitle>Use Cases</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-2 text-sm text-gray-600">
                  {dataset.use_cases.map((useCase, index) => (
                    <li key={index}>{useCase}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* User Feedback Card */}
            <Card>
              <CardHeader>
                <CardTitle>User Feedback</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dataset.user_feedback.comments.map((comment, index) => (
                    <div key={index} className="border-b pb-4 last:border-0 last:pb-0">
                      <p className="text-sm text-gray-600 mb-1">{comment.comment}</p>
                      <p className="text-xs text-gray-500">
                        {comment.user} • {new Date(comment.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* FAQs Card */}
            <Card>
              <CardHeader>
                <CardTitle>FAQs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dataset.user_feedback.faqs.map((faq, index) => (
                    <div key={index}>
                      <h4 className="font-medium text-sm mb-1">{faq.question}</h4>
                      <p className="text-sm text-gray-600">{faq.answer}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

// Wrap the export with Suspense
export default function DatasetDetailWrapper() {
  return (
    <Suspense fallback={<LoadingPlaceholder />}>
      <DatasetDetail />
    </Suspense>
  );
}

function LoadingPlaceholder() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="h-4 bg-gray-200 rounded w-full"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        </div>
      </div>
    </div>
  );
}
