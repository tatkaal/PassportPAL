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
    <div className="h-screen w-full bg-gradient-to-br from-primary-50 via-neutral-50 to-accent-50 animate-gradient overflow-hidden flex flex-col">
      <Toaster position="top-right" />
      
      {/* Redesigned header - more compact and aesthetic */}
      <header className="py-3 px-6 bg-white bg-opacity-80 shadow-sm animate-fade-in">
        <div className="flex items-center justify-between">
          <div className="flex flex-col">
            <h1 className="text-3xl font-bold text-primary-800 bg-clip-text text-transparent bg-gradient-to-r from-primary-700 to-primary-500">
              PassportPAL
            </h1>
            <p className="text-primary-600 text-sm">
              ID Document Classifier
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div
              className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium transition-all duration-300
                  ${
                    apiStatus === "online"
                      ? "bg-green-100 text-green-800"
                      : apiStatus === "offline"
                      ? "bg-red-100 text-red-800"
                      : "bg-yellow-100 text-yellow-800"
                  }`}
            >
              <span
                className={`h-2.5 w-2.5 mr-2 rounded-full ${
                  apiStatus === "online"
                    ? "bg-green-500 animate-pulse"
                    : apiStatus === "offline"
                    ? "bg-red-500"
                    : "bg-yellow-500"
                }`}
              />
              {apiStatus === "checking" ? "Connecting..." : `Backend API ${apiStatus}`}
            </div>
            
            <a 
              href="https://github.com/tatkaal/PassportPAL.git" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="inline-flex items-center transition-colors duration-300 text-primary-700 hover:text-primary-900 bg-neutral-100 px-3 py-1 rounded-full hover:bg-neutral-200"
            >
              <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.729.083-.729 1.205.085 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 21.795 24 17.295 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Main content area - strict height control to prevent overflow */}
      <div className="flex-1 flex gap-2 p-2 animate-slide-up overflow-hidden">
        {/* LEFT COLUMN - Sample Images (8% width) */}
        <div className="w-[8%] bg-white rounded-xl shadow-md p-3 flex flex-col">
          <h3 className="text-lg font-bold text-primary-700 mb-2 flex items-center">
          <svg className="w-5 h-5 mr-1.5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Samples
          </h3>
          <div className="pt-1 flex flex-col space-y-2 overflow-y-auto scrollbar-thin scrollbar-thumb-primary-300 flex-1">
            {SAMPLE_IMAGES.map((imgPath, index) => (
              <button
                key={index}
                onClick={() => handleSelectSampleImage(imgPath)}
                disabled={loading}
                className="flex-shrink-0 relative w-full h-20 rounded-lg border-2 hover:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 overflow-hidden bg-neutral-100 transition-all duration-300 transform hover:scale-105 active:scale-95"
              >
                {thumbnails[index] ? (
                  <img
                    src={thumbnails[index]}
                    alt={`Sample ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="flex items-center justify-center h-full w-full text-neutral-400">
                    <span className="text-xs">#{index + 1}</span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* MIDDLE AREA - Main Content (74% width) as a flex row */}
        <div className="w-[74%] flex flex-row gap-2 h-full">
          {/* Upload Document Box */}
          <div className="flex-1 bg-white rounded-xl shadow-md p-3 transition-all duration-300 hover:shadow-lg flex flex-col overflow-hidden">
            <h2 className="text-lg font-bold text-primary-700 mb-2 flex items-center">
              <svg className="w-5 h-5 mr-1.5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Upload Document
            </h2>

            {/* Dropzone - Reduced padding for compact display */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-2 flex-grow flex flex-col justify-between relative transition-all duration-300 overflow-hidden
                ${
                  isDragActive
                    ? "border-primary-500 bg-primary-50"
                    : "border-gray-300 hover:border-primary-500"
                }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <div className="flex flex-col justify-between h-full w-full">
                  {/* Center container with fixed height for consistent alignment */}
                  <div className="flex-1 flex items-center justify-center h-[calc(100%-48px)]">
                    <img
                      src={preview}
                      alt="Preview"
                      className="mx-auto rounded-lg shadow-md object-contain max-h-[75vh] w-auto max-w-[95%] animate-scale"
                    />
                  </div>
                  
                  {/* Fixed position button container at bottom */}
                  <div className="h-12 flex justify-center space-x-3 w-full">
                    <button
                      onClick={handleSubmit}
                      disabled={loading || apiStatus !== "online"}
                      className="px-4 py-1.5 bg-gradient-to-r from-primary-600 to-primary-700 text-white rounded-lg hover:from-primary-700 hover:to-primary-800 transition-all duration-300 shadow-md hover:shadow disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 active:scale-95"
                    >
                      {loading ? (
                        <span className="flex items-center">
                          <svg
                            className="animate-spin -ml-1 mr-2 h-3 w-3 text-white"
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
                      className="px-4 py-1.5 bg-neutral-200 text-neutral-700 rounded-lg hover:bg-neutral-300 transition-all duration-300 shadow-md hover:shadow disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 active:scale-95"
                    >
                      New Image
                    </button>
                  </div>
                </div>
              ) : (
                <div className="h-full flex items-center justify-center">
                  <div className="space-y-3 text-center">
                    <svg
                      className="mx-auto h-12 w-12 text-primary-400 animate-pulse-slow"
                      stroke="currentColor"
                      fill="none"
                      viewBox="0 0 48 48"
                    >
                      <path
                        d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 
                           01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 
                           0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 
                           0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                        strokeWidth={2.5}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    <p className="text-neutral-600">
                      {isDragActive
                        ? "Drop the image here..."
                        : "Drag & drop an ID document here, or click to browse"}
                    </p>
                    <p className="text-xs text-neutral-500">Supports JPEG, PNG</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Analysis Results Box */}
          <div className="flex-1 bg-white rounded-xl shadow-md p-3 transition-all duration-300 hover:shadow-lg flex flex-col overflow-hidden">
            <h2 className="text-lg font-bold text-primary-700 mb-2 flex items-center">
              <svg className="w-5 h-5 mr-1.5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
              </svg>
              Analysis Results
            </h2>
            {/* Match the structure exactly with the upload box for perfect alignment */}
            <div className="border-2 border-transparent rounded-lg p-2 flex-grow flex flex-col justify-between relative overflow-hidden">
              <div className="flex flex-col justify-between h-full w-full">
                <div className="flex-1 flex items-center justify-center h-[calc(100%-48px)]">
                  {result && result.segmentation ? (
                    <img
                      src={`data:image/jpeg;base64,${result.segmentation}`}
                      alt="Segmentation Result"
                      className="mx-auto rounded-lg shadow-md object-contain max-h-[75vh] w-auto max-w-[95%] animate-scale analysis-result-image"
                    />
                  ) : loading ? (
                    <div className="text-center">
                      <div className="relative w-16 h-16 mx-auto">
                        <div className="absolute top-0 w-full h-full rounded-full border-4 border-t-4 border-primary-500 border-t-transparent animate-spin"></div>
                        <div className="absolute top-0 w-full h-full rounded-full border-4 border-t-4 border-primary-300 border-opacity-50 border-t-transparent animate-spin" style={{ animationDuration: '1.5s' }}></div>
                      </div>
                      <p className="mt-3 text-primary-600">
                        Processing document...
                      </p>
                    </div>
                  ) : (
                    <div className="text-center text-neutral-500 flex flex-col items-center justify-center">
                      <svg
                        className="h-16 w-16 text-neutral-300 mb-2 animate-bounce-slow"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 
                          01-2-2V5a2 2 0 012-2h5.586a1 1 0 
                          01.707.293l5.414 5.414a1 1 0 
                          01.293.707V19a2 2 0 
                          01-2 2z"
                        />
                      </svg>
                      <p className="text-sm">Upload an image to view analysis results</p>
                    </div>
                  )}
                </div>
                {/* Empty div to match height with buttons on the left side */}
                <div className="h-12 w-full"></div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN - Predictions (18% width) */}
        <div className="w-[18%] bg-white rounded-xl shadow-md p-3 flex flex-col">
          <h3 className="text-base font-bold text-primary-700 mb-2 flex items-center">
            <svg className="w-5 h-5 mr-1.5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Predictions
          </h3>
          <div className="flex-grow overflow-y-auto scrollbar-thin scrollbar-thumb-primary-300">
            {result && result.top3_predictions ? (
              <div className="space-y-4">
                {result.top3_predictions.map((prediction, index) => (
                  <div key={index} className="transform transition-all duration-300 hover:translate-x-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-primary-800 flex items-center">
                        {index === 0 && <span className="text-xl mr-1">🥇</span>}
                        {index === 1 && <span className="text-xl mr-1">🥈</span>}
                        {index === 2 && <span className="text-xl mr-1">🥉</span>}
                        {prediction.class}
                      </span>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-white bg-primary-600 px-2 py-0.5 rounded-full">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-1 w-full bg-neutral-200 rounded-full h-2">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-primary-500 to-primary-700 transition-all duration-500 ease-out"
                        style={{
                          width: `${(prediction.confidence * 100).toFixed(1)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-neutral-400 italic text-xs">
                {loading ? "Analyzing..." : "No predictions yet"}
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="bg-primary-800 text-white py-2 text-center shadow-md animate-fade-in">
        <div className="container mx-auto">
          <p className="text-center text-sm">
            PassportPAL &copy; {new Date().getFullYear()} - ID Document Analysis Tool
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
