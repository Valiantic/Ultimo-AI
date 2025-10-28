/**
 * Main Page Component - RAG Chatbot Interface
 * 
 * This is the main chat interface where users:
 * 1. Upload PDF documents
 * 2. Ask questions about the documents
 * 3. Receive AI-generated answers
 * 
 * Component Structure:
 * - File upload section
 * - Chat messages display
 * - Input field for questions
 * - Loading states
 * - Error handling
 */

'use client'

import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Upload, Send, FileText, Loader2, AlertCircle } from 'lucide-react'

// Get API URL from environment variable
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Type definitions for better TypeScript support
interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
}

interface UploadResponse {
  message: string
  filename: string
  chunks_created: number
  collection_name: string
}

interface QueryResponse {
  answer: string
  sources: string[]
  confidence?: number
}

export default function Home() {
  // ==================== STATE MANAGEMENT ====================
  
  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  
  // Error state
  const [error, setError] = useState<string | null>(null)
  
  // Refs for auto-scrolling and file input
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ==================== EFFECTS ====================
  
  /**
   * Auto-scroll to bottom when new messages arrive
   */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ==================== FILE UPLOAD HANDLERS ====================
  
  /**
   * Handle file selection
   * Validates file type and size
   */
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    
    if (file) {
      // Validate file type
      if (file.type !== 'application/pdf') {
        setError('Please select a PDF file')
        return
      }
      
      // Validate file size (max 10MB)
      const maxSize = 10 * 1024 * 1024 // 10MB in bytes
      if (file.size > maxSize) {
        setError('File size must be less than 10MB')
        return
      }
      
      setSelectedFile(file)
      setError(null)
    }
  }

  /**
   * Upload PDF to backend
   * Sends file via FormData to /upload-pdf endpoint
   */
  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first')
      return
    }

    console.log('📤 FRONTEND: Starting upload...', {
      fileName: selectedFile.name,
      fileSize: `${(selectedFile.size / 1024 / 1024).toFixed(2)}MB`,
      fileType: selectedFile.type
    })

    setIsUploading(true)
    setError(null)

    try {
      // Create FormData to send file
      console.log('📦 FRONTEND: Creating FormData...')
      const formData = new FormData()
      formData.append('file', selectedFile)
      console.log('✅ FRONTEND: FormData created')

      // Send POST request to backend
      console.log(`🌐 FRONTEND: Sending POST to ${API_URL}/upload-pdf`)
      console.log('⏳ FRONTEND: Waiting for backend response...')
      
      const response = await axios.post<UploadResponse>(
        `${API_URL}/upload-pdf`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = progressEvent.total 
              ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
              : 0
            console.log(`📊 FRONTEND: Upload progress: ${percentCompleted}%`)
          }
        }
      )

      console.log('✅ FRONTEND: Upload successful!', response.data)

      // Show success message
      setUploadSuccess(true)
      setMessages([
        {
          role: 'assistant',
          content: `✅ ${response.data.message}\n\nFile: ${response.data.filename}\nChunks created: ${response.data.chunks_created}\n\nYou can now ask questions about this document!`,
        },
      ])

      // Reset file input
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      console.log('🎉 FRONTEND: Upload process complete!')

    } catch (err: any) {
      console.error('❌ FRONTEND: Upload error:', err)
      console.error('❌ FRONTEND: Error details:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status
      })
      
      setError(
        err.response?.data?.detail || 
        'Failed to upload file. Please try again.'
      )
    } finally {
      setIsUploading(false)
      console.log('🏁 FRONTEND: Upload state reset')
    }
  }

  // ==================== CHAT HANDLERS ====================
  
  /**
   * Send question to backend
   * Handles the RAG query process
   */
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return
    
    if (!uploadSuccess) {
      setError('Please upload a PDF first')
      return
    }

    const userMessage = inputMessage.trim()
    console.log('💬 FRONTEND: Sending query:', userMessage)
    
    setInputMessage('')
    setIsLoading(true)
    setError(null)

    // Add user message to chat
    const newUserMessage: Message = {
      role: 'user',
      content: userMessage,
    }
    setMessages((prev) => [...prev, newUserMessage])

    try {
      // Prepare chat history for context
      const chatHistory = messages
        .filter((m) => m.role !== 'assistant' || !m.content.includes('✅'))
        .map((m) => [m.content, m.role === 'user' ? '' : m.content])

      console.log('🔍 FRONTEND: Prepared chat history:', chatHistory.length, 'messages')

      // Send query to backend
      console.log(`🌐 FRONTEND: Sending query to ${API_URL}/query`)
      const response = await axios.post<QueryResponse>(
        `${API_URL}/query`,
        {
          question: userMessage,
          chat_history: chatHistory,
        }
      )

      console.log('✅ FRONTEND: Query successful!', {
        answerLength: response.data.answer.length,
        sourcesCount: response.data.sources?.length || 0
      })

      // Add assistant response to chat
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
      }
      setMessages((prev) => [...prev, assistantMessage])

    } catch (err: any) {
      console.error('❌ FRONTEND: Query error:', err)
      console.error('❌ FRONTEND: Error details:', {
        message: err.message,
        response: err.response?.data,
        status: err.response?.status
      })
      
      setError(
        err.response?.data?.detail || 
        'Failed to get response. Please try again.'
      )
    } finally {
      setIsLoading(false)
      console.log('🏁 FRONTEND: Query state reset')
    }
  }

  /**
   * Handle Enter key press in input
   */
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // ==================== RENDER ====================
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-4">
      <div className="max-w-4xl mx-auto space-y-6 py-8">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-white">
            RAG Chatbot
          </h1>
          <p className="text-slate-600 dark:text-slate-300">
            Upload a PDF and ask questions about its content
          </p>
        </div>

        {/* File Upload Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Upload Document
            </CardTitle>
            <CardDescription>
              Select a PDF file (max 10MB) to analyze
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* File input */}
            <div className="flex gap-2">
              <Input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                disabled={isUploading}
                className="cursor-pointer"
              />
              <Button
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className="min-w-[100px]"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Uploading
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload
                  </>
                )}
              </Button>
            </div>

            {/* Error display */}
            {error && (
              <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950 p-3 rounded-md">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Chat Card */}
        <Card className="flex flex-col h-[500px]">
          <CardHeader>
            <CardTitle>Chat</CardTitle>
            <CardDescription>
              Ask questions about your uploaded document
            </CardDescription>
          </CardHeader>
          
          <CardContent className="flex-1 flex flex-col p-0">
            {/* Messages Display */}
            <ScrollArea className="flex-1 p-4 scrollbar-thin max-h-[400px] overflow-y-auto">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center h-full text-slate-400">
                  <div className="text-center space-y-2">
                    <FileText className="w-12 h-12 mx-auto opacity-50" />
                    <p>No messages yet</p>
                    <p className="text-sm">Upload a PDF to start chatting</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${
                        message.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg p-4 ${
                          message.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted'
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        
                        {/* Display sources if available */}
                        {message.sources && message.sources.length > 0 && (
                          <details className="mt-3 text-sm opacity-80">
                            <summary className="cursor-pointer font-semibold">
                              Sources ({message.sources.length})
                            </summary>
                            <div className="mt-2 space-y-2">
                              {message.sources.map((source, idx) => (
                                <div
                                  key={idx}
                                  className="p-2 bg-background/50 rounded border border-border text-xs"
                                >
                                  {source.substring(0, 200)}...
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    </div>
                  ))}
                  
                  {/* Loading indicator */}
                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="bg-muted rounded-lg p-4">
                        <Loader2 className="w-5 h-5 animate-spin" />
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>
              )}
            </ScrollArea>

            {/* Input Area */}
            <div className="p-4 border-t">
              <div className="flex gap-2">
                <Input
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={
                    uploadSuccess
                      ? "Ask a question about your document..."
                      : "Upload a PDF first..."
                  }
                  disabled={!uploadSuccess || isLoading}
                  className="flex-1"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!uploadSuccess || isLoading || !inputMessage.trim()}
                  size="icon"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center text-sm text-slate-500">
          <p>Powered by FastAPI, Qdrant, LangChain & Google Gemini</p>
        </div>
      </div>
    </div>
  )
}