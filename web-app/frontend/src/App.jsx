import { useState, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { Toaster, toast } from "react-hot-toast";

// Sample image paths
const SAMPLE_IMAGES = [
  "/samples/sample1.jpg",
  "/samples/sample2.jpg",
  "/samples/sample3.jpg",
  "/samples/sample4.jpg",
  "/samples/sample5.jpg",
  "/samples/sample6.jpg",
  "/samples/sample7.jpg",
];

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [thumbnails, setThumbnails] = useState([]);

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await axios.get("/api/status");
        if (response.data.status === "online") {
          setApiStatus("online");
          loadThumbnails();
        } else {
          setApiStatus("offline");
        }
      } catch (error) {
        setApiStatus("offline");
      }
    };

    const loadThumbnails = async () => {
      const loadedUrls = await Promise.all(
        SAMPLE_IMAGES.map(async (path) => {
          try {
            const res = await fetch(
              `/api/get-sample?path=${encodeURIComponent(
                path
              )}`,
              { method: "GET" }
            );
            if (!res.ok) return null;
            const blob = await res.blob();
            return URL.createObjectURL(blob);
          } catch (error) {
            console.warn("Failed to load thumbnail:", error);
            return null;
          }
        })
      );
      setThumbnails(loadedUrls);
    };

    checkApiStatus();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      setResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpeg", ".jpg", ".png"] },
    maxFiles: 1,
    noClick: !!preview,
    noKeyboard: !!preview,
  });

  const handleNewImage = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    open();
  };

  const handleSelectSampleImage = async (imagePath) => {
    try {
      setLoading(true);
      const apiUrl = `/api/get-sample?path=${encodeURIComponent(
        imagePath
      )}`;
      const response = await fetch(apiUrl);
      if (!response.ok) {
        throw new Error(`Failed to load sample image: ${response.statusText}`);
      }
      const blob = await response.blob();
      const fileName = imagePath.split("/").pop() || "sample.jpg";
      const file = new File([blob], fileName, { type: blob.type });
      setImage(file);

      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      setResult(null);
    } catch (error) {
      toast.error("Failed to load sample image: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      toast.error("Please upload an image first");
      return;
    }
    if (apiStatus !== "online") {
      toast.error("Backend API is not available.");
      return;
    }
    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post(
        "/api/analyze",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(response.data);
      toast.success("Analysis completed!");
    } catch (error) {
      toast.error(
        `Error analyzing image: ${
          error.response?.data?.detail || error.message
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-primary-50 to-primary-100">
      <Toaster position="top-right" />
      <div className="container mx-auto py-6 md:py-10 px-4 md:px-6 max-w-7xl">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-center text-primary-800 mb-2">
            PassportPAL - ID Document Classifier
          </h1>
          <p className="text-center text-primary-600">
            Upload an ID document to detect and classify it using simple CNN classifier
          </p>
          <div className="flex justify-center mt-2">
            <div
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                  ${
                    apiStatus === "online"
                      ? "bg-green-100 text-green-800"
                      : apiStatus === "offline"
                      ? "bg-red-100 text-red-800"
                      : "bg-yellow-100 text-yellow-800"
                  }`}
            >
              <span
                className={`h-2 w-2 mr-1 rounded-full 
                    ${
                      apiStatus === "online"
                        ? "bg-green-400"
                        : apiStatus === "offline"
                        ? "bg-red-400"
                        : "bg-yellow-400"
                    }`}
              />
              {apiStatus === "checking" ? "Connecting..." : `API ${apiStatus}`}
            </div>
          </div>
        </header>

        {/* Layout: stack on small, 2-col on large */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
          {/* LEFT PANEL */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-primary-800 mb-4">
              Upload Document
            </h2>

            {/* Sample Images */}
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">
                Try a sample image:
              </h3>
              <div className="flex space-x-2 overflow-x-auto pb-2">
                {SAMPLE_IMAGES.map((imgPath, index) => (
                  <button
                    key={index}
                    onClick={() => handleSelectSampleImage(imgPath)}
                    disabled={loading}
                    className="flex-shrink-0 relative w-20 h-20 rounded border-2 hover:border-primary-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 overflow-hidden bg-gray-100"
                  >
                    {thumbnails[index] ? (
                      <img
                        src={thumbnails[index]}
                        alt={`Sample ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full w-full text-gray-400">
                        <span className="text-xs">#{index + 1}</span>
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                ${
                  isDragActive
                    ? "border-primary-500 bg-primary-50"
                    : "border-gray-300 hover:border-primary-500"
                }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <div>
                  {/* Show the preview bigger for portrait images */}
                  <img
                    src={preview}
                    alt="Preview"
                    className="mx-auto rounded-lg shadow-md"
                    style={{
                      maxHeight: "500px", // bigger for portrait
                      width: "auto",
                      maxWidth: "100%",
                    }}
                  />
                  <div className="mt-4 flex justify-center space-x-4">
                    <button
                      onClick={handleSubmit}
                      disabled={loading || apiStatus !== "online"}
                      className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? (
                        <span className="flex items-center">
                          <svg
                            className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            />
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 
                                 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            />
                          </svg>
                          Analyzing...
                        </span>
                      ) : (
                        "Analyze Image"
                      )}
                    </button>
                    <button
                      onClick={handleNewImage}
                      disabled={loading}
                      className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      New Image
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 
                         01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 
                         0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 
                         0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <p className="text-gray-600">
                    {isDragActive
                      ? "Drop the image here..."
                      : "Drag & drop an ID document here, or click to browse"}
                  </p>
                  <p className="text-sm text-gray-500">Supports JPEG, PNG</p>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT PANEL */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-primary-800 mb-4">
              Analysis Results
            </h2>
            {result ? (
              <div className="space-y-4">
                {result.segmentation && (
                  <div className="p-4 bg-gradient-to-r from-primary-50 to-primary-100 rounded-lg">
                    <h3 className="font-medium text-primary-800 mb-2">
                      Detected Document
                    </h3>
                    <div className="relative">
                      <img
                        src={`data:image/jpeg;base64,${result.segmentation}`}
                        alt="Segmentation Result"
                        className="mx-auto rounded-lg shadow-md"
                        style={{
                          maxHeight: "700px", // bigger portrait
                          width: "auto",
                          maxWidth: "100%",
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Show classification results (top 3) in a more spacious UI */}
                <div className="p-4 bg-gradient-to-r from-primary-50 to-primary-100 rounded-lg">
                  <h3 className="font-medium text-primary-800 mb-2">
                    Predictions
                  </h3>
                  {result.top3_predictions &&
                    result.top3_predictions.map((prediction, index) => (
                      <div key={index} className="mb-3 last:mb-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium text-primary-800">
                            {index === 0 && "🥇 "}
                            {index === 1 && "🥈 "}
                            {index === 2 && "🥉 "}
                            {prediction.class}
                          </span>
                          <span className="text-sm font-semibold text-primary-600">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div
                            className="h-2.5 rounded-full bg-primary-600"
                            style={{
                              width: `${(prediction.confidence * 100).toFixed(
                                1
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ) : loading ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
                  <p className="mt-4 text-primary-600">
                    Processing document...
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-16">
                <svg
                  className="mx-auto h-16 w-16 text-gray-300"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 
                       01-2-2V5a2 2 0 012-2h5.586a1 1 0 
                       01.707.293l5.414 5.414a1 1 0 
                       01.293.707V19a2 2 0 
                       01-2 2z"
                  />
                </svg>
                <p className="mt-4">Upload an image to view analysis results</p>
              </div>
            )}
          </div>
        </div>

        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>
            PassportPAL &copy; {new Date().getFullYear()} - ID Document Analysis Tool
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
