import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Upload,
  FileType,
  Tag,
  Info,
  Plus,
  X,
  AlertCircle,
  ChevronLeft
} from "lucide-react";
import { Link } from "react-router-dom";
import dummyData from "../data/dummyData.json";

const DatasetUpload = () => {
  const { id } = useParams();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState([]);
  const [tags, setTags] = useState([]);
  const [currentTag, setCurrentTag] = useState("");
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    // Load existing dataset if editing
    if (id) {
      const dataset = dummyData.datasets.find(d => d.id === id);
      if (dataset) {
        setTitle(dataset.title);
        setDescription(dataset.description);
        setTags(dataset.tags);
        setFiles(dataset.files.map(f => ({
          name: f.name,
          size: f.size,
          type: f.type
        })));
      }
    }
  }, [id]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFiles = (uploadedFiles) => {
    setFiles([...files, ...Array.from(uploadedFiles)]);
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleTagAdd = (e) => {
    e.preventDefault();
    if (currentTag && !tags.includes(currentTag)) {
      setTags([...tags, currentTag]);
      setCurrentTag("");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 py-8">
      <div className="max-w-5xl mx-auto px-4">
        {/* Back Button */}
        <Link 
          to="/datasets" 
          className="inline-flex items-center text-slate-600 hover:text-blue-600 transition-colors duration-200 mb-8 group"
        >
          <ChevronLeft className="h-5 w-5 mr-1 group-hover:-translate-x-1 transition-transform duration-200" />
          Back to Datasets
        </Link>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-xl shadow-lg border border-slate-100 p-8"
        >
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-2">
            {id ? "Edit Dataset" : "Upload Dataset"}
          </h1>
          <p className="text-slate-600 mb-8">
            Share your valuable dataset with the research community
          </p>

          <form className="space-y-8" onSubmit={(e) => e.preventDefault()}>
            {/* Basic Info Section */}
            <div className="space-y-6">
              <motion.div 
                whileHover={{ scale: 1.01 }}
                className="transition-all duration-200"
              >
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Dataset Title
                </label>
                <Input 
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter a descriptive title for your dataset"
                  className="w-full border-slate-200 focus:ring-2 focus:ring-blue-500 transition-all duration-200"
                />
              </motion.div>

              <motion.div 
                whileHover={{ scale: 1.01 }}
                className="transition-all duration-200"
              >
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Description
                </label>
                <Textarea 
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe your dataset, its contents, and potential use cases..."
                  className="w-full h-32 border-slate-200 focus:ring-2 focus:ring-blue-500 transition-all duration-200"
                />
              </motion.div>

              {/* Enhanced Tags Section */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Tags
                </label>
                <div className="flex flex-wrap gap-2 mb-3">
                  {tags.map((tag, index) => (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      key={index}
                    >
                      <Badge className="bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors duration-200 gap-1 px-3 py-1">
                        {tag}
                        <X 
                          className="h-3 w-3 cursor-pointer hover:text-blue-800" 
                          onClick={() => setTags(tags.filter((_, i) => i !== index))}
                        />
                      </Badge>
                    </motion.div>
                  ))}
                </div>
                <div className="flex gap-2">
                  <Input
                    placeholder="Add relevant tags..."
                    value={currentTag}
                    onChange={(e) => setCurrentTag(e.target.value)}
                    className="border-slate-200 focus:ring-2 focus:ring-blue-500"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleTagAdd(e);
                      }
                    }}
                  />
                  <Button 
                    type="button"
                    variant="outline" 
                    size="icon"
                    onClick={handleTagAdd}
                    className="hover:bg-blue-50 hover:text-blue-600 transition-colors duration-200"
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Enhanced File Upload Zone */}
            <motion.div
              whileHover={{ scale: 1.01 }}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
                dragActive 
                  ? "border-blue-500 bg-blue-50" 
                  : "border-slate-200 hover:border-blue-500"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="flex flex-col items-center">
                <Upload className="h-12 w-12 text-blue-500 mb-4" />
                <p className="text-slate-700 text-lg mb-2">
                  Drag and drop your files here, or{" "}
                  <label className="text-blue-600 hover:text-blue-700 cursor-pointer underline decoration-dashed underline-offset-4">
                    browse
                    <input
                      type="file"
                      multiple
                      className="hidden"
                      onChange={(e) => handleFiles(e.target.files)}
                    />
                  </label>
                </p>
                <p className="text-sm text-slate-500">
                  Supported formats: CSV, JSON, Excel, ZIP (max 50GB)
                </p>
              </div>
            </motion.div>

            {/* Enhanced File List */}
            {files.length > 0 && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-3"
              >
                <h3 className="text-sm font-medium text-slate-700">
                  Uploaded Files ({files.length})
                </h3>
                <div className="space-y-2">
                  {files.map((file, index) => (
                    <motion.div 
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center justify-between p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors duration-200"
                    >
                      <div className="flex items-center space-x-3">
                        <FileType className="h-6 w-6 text-blue-500" />
                        <div>
                          <p className="text-sm font-medium text-slate-700">
                            {file.name}
                          </p>
                          <p className="text-xs text-slate-500">
                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => removeFile(index)}
                        className="hover:bg-red-50 hover:text-red-500 transition-colors duration-200"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Enhanced License Info */}
            <motion.div 
              whileHover={{ scale: 1.01 }}
              className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-100"
            >
              <div className="flex items-start space-x-3">
                <Info className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-blue-900 mb-1">
                    Dataset License
                  </h4>
                  <p className="text-sm text-blue-700">
                    By uploading, you agree to share this dataset under the Creative Commons License (CC BY-SA 4.0)
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Enhanced Submit Button */}
            <div className="flex justify-end">
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button 
                  type="submit" 
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-2 text-lg shadow-lg hover:shadow-xl transition-all duration-200"
                >
                  <Upload className="h-5 w-5 mr-2" />
                  {id ? "Update Dataset" : "Upload Dataset"}
                </Button>
              </motion.div>
            </div>
          </form>
        </motion.div>
      </div>
    </div>
  );
};

export default DatasetUpload;
