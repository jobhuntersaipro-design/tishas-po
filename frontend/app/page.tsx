"use client";

import { useState, useEffect, useRef } from "react";
import { uploadFiles, uploadFileBatch, getBatchStatus, fetchHistory, saveAllPOs } from "@/lib/api";
import { PODocument } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { POCard as POCardComponent } from "@/components/po-card";
import { HistoryTable } from "@/components/history-table";
import { Toaster } from "@/components/ui/sonner";
import { Loader2, UploadCloud, X, Save, Mail } from "lucide-react";
import { toast } from "sonner";
import { formatNumber, formatQuantity } from "@/lib/utils";

interface FileWithStatus {
  file: File;
  status: 'pending' | 'processing' | 'completed' | 'saved' | 'error' | 'duplicate';
  doc?: PODocument;
  error?: string;
  progress?: string;
}

export default function Home() {
  const [filesWithStatus, setFilesWithStatus] = useState<FileWithStatus[]>([]);
  const [processedDocs, setProcessedDocs] = useState<PODocument[]>([]);
  const [historyData, setHistoryData] = useState<PODocument[]>([]);
  const [activeTab, setActiveTab] = useState("process");
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Load last visited tab from localStorage on mount (client-side only)
  useEffect(() => {
    const stored = localStorage.getItem('activeTab');
    if (stored) {
      setActiveTab(stored);
    }
  }, []);
  const [isSavingAll, setIsSavingAll] = useState(false);

  // Save active tab to localStorage when it changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('activeTab', activeTab);
    }
  }, [activeTab]);

  // Load history when tab changes
  useEffect(() => {
    if (activeTab === "history") {
      setIsLoadingHistory(true);
      fetchHistory()
        .then(setHistoryData)
        .finally(() => setIsLoadingHistory(false));
    }
  }, [activeTab]);

  const handleRefreshHistory = () => {
    setIsLoadingHistory(true);
    fetchHistory()
      .then(setHistoryData)
      .finally(() => setIsLoadingHistory(false));
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles: FileWithStatus[] = Array.from(e.target.files).map(file => ({
        file,
        status: 'pending' as const
      }));

      setFilesWithStatus((prev) => [...prev, ...newFiles]);

      // Reset input value to allow re-uploading the same file
      e.target.value = '';

      // Process files immediately
      processNewFiles(newFiles);
    }
  };

  // Ref to track current files for async operations to avoid zombie updates
  const filesRef = useRef<FileWithStatus[]>([]);
  useEffect(() => {
    filesRef.current = filesWithStatus;
  }, [filesWithStatus]);

  const processNewFiles = async (filesToProcess: FileWithStatus[]) => {
    for (const fileWithStatus of filesToProcess) {
      // Update status to processing
      setFilesWithStatus(prev =>
        prev.map(f => f.file === fileWithStatus.file
          ? { ...f, status: 'processing' as const, progress: `Starting upload...` }
          : f
        )
      );

      try {
        // Start batch processing
        const { batch_id } = await uploadFileBatch(fileWithStatus.file);

        // Track POs we've already added to avoid duplicates
        const seenPoNumbers = new Set<string>();

        // Poll for results
        const pollInterval = setInterval(async () => {
          try {
            const batchStatus = await getBatchStatus(batch_id);

            // Check if file is still in the active list
            const isFileStillActive = filesRef.current.some(f => f.file === fileWithStatus.file);
            if (!isFileStillActive) {
              clearInterval(pollInterval);
              return;
            }

            // Update progress message
            const { current, total } = batchStatus.progress;
            if (total > 0) {
              setFilesWithStatus(prev =>
                prev.map(f => f.file === fileWithStatus.file
                  ? { ...f, progress: `Processing page ${current}/${total}...` }
                  : f
                )
              );
            }

            // Add new POs incrementally
            for (const po of batchStatus.pos) {
              const poNumber = po.po_number || '';
              if (!seenPoNumbers.has(poNumber)) {
                seenPoNumbers.add(poNumber);

                // Check if already exists
                if (po.already_exists) {
                  toast.info(po.duplicate_message || `${po.po_number} already exists in database`);
                } else {
                  // Add to processed docs
                  setProcessedDocs(prev => [...prev, po]);
                  toast.success(`Extracted PO ${po.po_number || 'data'} from page ${current}`);
                }
              }
            }

            // Check if complete or error
            if (batchStatus.status === 'complete') {
              clearInterval(pollInterval);

              setFilesWithStatus(prev =>
                prev.map(f => f.file === fileWithStatus.file
                  ? { ...f, status: 'completed' as const, progress: undefined }
                  : f
                )
              );

              // Show page errors if any
              if (batchStatus.page_errors && batchStatus.page_errors.length > 0) {
                toast.warning(`Completed with ${batchStatus.page_errors.length} page error(s)`);
              }
            } else if (batchStatus.status === 'error') {
              clearInterval(pollInterval);

              const errorMessage = batchStatus.error || 'Processing failed';
              setFilesWithStatus(prev =>
                prev.map(f => f.file === fileWithStatus.file
                  ? { ...f, status: 'error' as const, error: errorMessage }
                  : f
                )
              );
              toast.error(`${fileWithStatus.file.name}: ${errorMessage}`);
            }
          } catch (pollError) {
            console.error('Polling error:', pollError);
            // Don't stop polling on transient errors
          }
        }, 1500); // Poll every 1.5 seconds

      } catch (error: any) {
        // Check if file is still active even on error
        const isFileStillActive = filesRef.current.some(f => f.file === fileWithStatus.file);
        if (!isFileStillActive) continue;

        let errorMessage = "Error processing file";
        if (error.message) {
          errorMessage = error.message;
        }

        setFilesWithStatus(prev =>
          prev.map(f => f.file === fileWithStatus.file
            ? { ...f, status: 'error' as const, error: errorMessage }
            : f
          )
        );
        toast.error(`${fileWithStatus.file.name}: ${errorMessage}`);
      }
    }
  };

  const handleSaveAll = async () => {
    if (processedDocs.length === 0) {
      toast.error("No POs to save");
      return;
    }

    setIsSavingAll(true);
    const result = await saveAllPOs(processedDocs);
    setIsSavingAll(false);

    if (result.success) {
      toast.success(result.message);

      // Update status of saved files to 'saved'
      const savedPoNumbers = processedDocs.map(d => d.po_number);
      setFilesWithStatus(prev => prev.map(f =>
        (f.doc && savedPoNumbers.includes(f.doc.po_number))
          ? { ...f, status: 'saved' as const }
          : f
      ));

      // Clear processed docs queue
      setProcessedDocs([]);

      // Refresh history if on that tab
      if (activeTab === "history") {
        fetchHistory().then(setHistoryData);
      }
    } else {
      toast.error(result.message);
    }
  };

  const removeFile = (index: number) => {
    setFilesWithStatus(prev => prev.filter((_, i) => i !== index));
  };

  const removeDoc = (index: number) => {
    setProcessedDocs(prev => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFilesWithStatus([]);
    setProcessedDocs([]);
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const newFiles: FileWithStatus[] = files.map(file => ({
        file,
        status: 'pending' as const,
      }));
      setFilesWithStatus(prev => [...prev, ...newFiles]);

      // Process files immediately (same as click upload)
      processNewFiles(newFiles);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1 container mx-auto py-10 px-4 md:px-6 lg:px-8">
        <Toaster />
        <div className="mb-8 text-center bg-gray-50 dark:bg-zinc-900 p-8 rounded-lg">
          <h1 className="text-2xl md:text-4xl font-extrabold tracking-tight lg:text-5xl mb-2">
            Tisha's PO Extractor
          </h1>
        </div>

        <Tabs defaultValue="process" value={activeTab} onValueChange={setActiveTab} className="space-y-3">
          <TabsList className="grid w-full grid-cols-2 h-14">
            <TabsTrigger value="process" className="text-md font-semibold">Process New POs</TabsTrigger>
            <TabsTrigger value="history" className="text-md font-semibold">Database History</TabsTrigger>
          </TabsList>

          {/* TAB 1: PROCESS */}
          <TabsContent value="process" className="space-y-4">
            <label htmlFor="file-upload-main" className="cursor-pointer block">
              <div
                className={`flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-10 transition-all ${isDragging
                  ? 'border-primary bg-primary/10 scale-[1.02]'
                  : 'border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-zinc-900/50 hover:bg-gray-100 dark:hover:bg-zinc-900/70'
                  }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <UploadCloud className={`h-12 w-12 mb-4 transition-colors ${isDragging ? 'text-primary' : 'text-primary'
                  }`} />
                <h3 className="text-xl font-bold mb-4 text-center">
                  {isDragging ? 'Drop files here' : 'Upload Purchase Order Files'}
                </h3>
                <div className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-bold text-md hover:bg-primary/90 transition-colors shadow-lg pointer-events-none">
                  Choose Files
                </div>
                <p className="mt-4 text-sm text-muted-foreground text-center">
                  {isDragging ? 'Release to upload' : 'Drag & drop files here or click to browse'}
                  <br />
                  <span className="text-xs">PDF, JPG, JPEG, or PNG files accepted</span>
                </p>
              </div>
            </label>
            <Input
              id="file-upload-main"
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png"
              className="hidden"
              onChange={handleFileChange}
            />


            {/* Show files being processed */}
            {filesWithStatus.length > 0 && (
              <div className="mt-6">
                <h3 className="text-xl font-bold mb-3">Uploaded Files ({filesWithStatus.length})</h3>
                <div className="space-y-2">
                  {filesWithStatus.map((fileWithStatus, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-zinc-900/50 rounded-lg">
                      <div className="flex items-center gap-3 flex-1">
                        <div className="flex-1">
                          <p className="font-medium text-sm">{fileWithStatus.file.name}</p>
                          {fileWithStatus.status === 'processing' && fileWithStatus.progress && (
                            <p className="text-xs text-muted-foreground mt-1">{fileWithStatus.progress}</p>
                          )}
                          {fileWithStatus.status === 'completed' && (
                            <p className="text-xs text-green-600 dark:text-green-400 mt-1">✓ Extraction complete</p>
                          )}
                          {fileWithStatus.status === 'duplicate' && (
                            <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">⚠ {fileWithStatus.error}</p>
                          )}
                          {fileWithStatus.status === 'error' && (
                            <p className="text-xs text-red-600 dark:text-red-400 mt-1">✗ {fileWithStatus.error}</p>
                          )}
                          {fileWithStatus.status === 'processing' && (
                            <p className="text-xs text-muted-foreground flex items-center gap-1">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Processing...
                            </p>
                          )}
                          {fileWithStatus.status === 'completed' && (
                            <p className="text-xs text-green-600 dark:text-green-400">
                              ✓ Processed successfully
                            </p>
                          )}
                          {fileWithStatus.status === 'saved' && (
                            <p className="text-xs text-green-600 dark:text-blue-400 font-medium">
                              ✓ Saved successfully
                            </p>
                          )}
                          {fileWithStatus.status === 'duplicate' && (
                            <p className="text-xs text-orange-600 dark:text-orange-400">
                              ⚠ Already exists in database
                            </p>
                          )}
                          {fileWithStatus.status === 'error' && (
                            <p className="text-xs text-red-600 dark:text-red-400">
                              ✗ {fileWithStatus.error}
                            </p>
                          )}
                          {fileWithStatus.status === 'pending' && (
                            <p className="text-xs text-muted-foreground">
                              Waiting...
                            </p>
                          )}
                        </div>
                      </div>
                      {/* Show X button for errors, duplicates, processing, pending, or completed (not yet saved) */}
                      {fileWithStatus.status !== 'saved' && (
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeFile(idx)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {(filesWithStatus.length > 0 || processedDocs.length > 0) && (
              <div className="flex gap-2 mt-4">
                {processedDocs.length > 0 && (
                  <Button
                    onClick={handleSaveAll}
                    disabled={isSavingAll}
                    size="lg"
                    className="font-bold"
                  >
                    {isSavingAll ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="mr-2 h-5 w-5" />
                        Save all ({processedDocs.length} POs) to database
                      </>
                    )}
                  </Button>
                )}
                <Button
                  variant="outline"
                  onClick={clearAll}
                >
                  Clear All
                </Button>
              </div>
            )}


            {/* Show processed documents for editing */}
            {processedDocs.length > 0 && (
              <div className="mt-8">
                <h3 className="text-2xl font-bold mb-4">Extracted Results ({processedDocs.length})</h3>
                <div className="space-y-8">
                  {processedDocs.map((doc, idx) => (
                    <POCardComponent
                      key={idx}
                      doc={doc}
                      onSaved={() => removeDoc(idx)}
                      onRemove={() => removeDoc(idx)}
                    />
                  ))}
                </div>
              </div>
            )}

            {filesWithStatus.length === 0 && processedDocs.length === 0 && (
              <div className="text-center py-20 text-muted-foreground">
                No documents to process. Upload files above.
              </div>
            )}
          </TabsContent>

          {/* TAB 2: HISTORY */}
          <TabsContent value="history">
            {isLoadingHistory ? (
              <div className="flex flex-col items-center justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
                <p className="text-muted-foreground">Loading history from database...</p>
              </div>
            ) : (
              <HistoryTable
                data={historyData}
                onRefresh={handleRefreshHistory}
                isRefreshing={isLoadingHistory}
              />
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <footer className="border-t mt-auto bg-gray-50 dark:bg-zinc-900/50">
        <div className="container mx-auto py-6">
          <div className="flex flex-col md:flex-row items-center justify-center gap-1 md:gap-2 text-sm text-muted-foreground">
            <span>Powered by <span className="font-semibold">Kim Brothers Ent.</span></span>
            <span className="hidden md:inline">|</span>
            <a
              href="mailto:jobhunters.ai.pro@gmail.com"
              className="flex items-center gap-1 hover:text-primary transition-colors"
            >
              <Mail className="h-4 w-4" />
              jobhunters.ai.pro@gmail.com
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
