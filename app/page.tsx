"use client"
import { useState } from "react"
import type React from "react"

import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import {
  Target,
  Database,
  Brain,
  Rocket,
  RotateCcw,
  BarChart3,
  Code,
  Zap,
  FileCode,
  CheckCircle,
  Play,
  ChevronLeft,
  ChevronRight,
  Send,
  ExternalLink,
} from "lucide-react"

// Module 1 structure matching the screenshot
const module1Sections = [
  {
    id: "problem-definition",
    title: "1. Problem Definition & Goal Setting",
    icon: Target,
    completed: false,
  },
  {
    id: "data-collection",
    title: "2. Data Collection & Preparation",
    icon: Database,
    completed: false,
  },
  {
    id: "model-training",
    title: "3. Model Training & Evaluation",
    icon: Brain,
    completed: false,
  },
  {
    id: "model-deployment",
    title: "4. Model Deployment & Monitoring",
    icon: Rocket,
    completed: false,
  },
  {
    id: "iteration",
    title: "5. Iteration & Improvement",
    icon: RotateCcw,
    completed: false,
  },
  {
    id: "interactive-lab-comparison",
    title: "Interactive Lab: Model Comparison Dashboard",
    icon: BarChart3,
    completed: false,
  },
  {
    id: "interactive-lab-sensitivity",
    title: "Interactive Lab: Metric Sensitivity Explorer",
    icon: Zap,
    completed: false,
  },
  {
    id: "interactive-lab-optimization",
    title: "Interactive Lab: Hyperparameter Optimization Tracker",
    icon: Target,
    completed: false,
  },
  {
    id: "interactive-lab-explainability",
    title: "Interactive Lab: Explainability Explorer",
    icon: Brain,
    completed: false,
  },
  {
    id: "coding-assignment",
    title: "Coding Assignment: ML Model Development",
    icon: Code,
    completed: false,
  },
  {
    id: "mini-project",
    title: "Mini-Project: Deploying a Fraud Detection Model",
    icon: FileCode,
    completed: false,
  },
]

export default function Module1Platform() {
  const [activeSection, setActiveSection] = useState("problem-definition")
  const [completedSections, setCompletedSections] = useState<string[]>([])
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const markComplete = () => {
    if (!completedSections.includes(activeSection)) {
      setCompletedSections([...completedSections, activeSection])
    }
  }

  const currentSection = module1Sections.find((s) => s.id === activeSection)
  const completedCount = completedSections.length
  const totalSections = module1Sections.length
  const progressPercentage = (completedCount / totalSections) * 100

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Left Sidebar */}
        <div
          className={`${
            sidebarCollapsed ? "w-16" : "w-80"
          } bg-white border-r border-gray-200 min-h-screen transition-all duration-300 relative`}
        >
          {/* Collapse Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="absolute -right-3 top-6 z-10 bg-white border border-gray-200 rounded-full p-1 h-6 w-6"
          >
            {sidebarCollapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
          </Button>

          <div className="p-6">
            {!sidebarCollapsed && (
              <>
                <h1 className="text-xl font-bold text-gray-900 mb-2">Module 1: Machine Learning Fundamentals</h1>
                <p className="text-sm text-gray-600 mb-4">
                  {completedCount} of {totalSections} sections completed
                </p>

                {/* Progress Bar */}
                <div className="mb-6">
                  <Progress value={progressPercentage} className="h-2" />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>
                      {completedCount}/{totalSections} Complete
                    </span>
                  </div>
                </div>
              </>
            )}

            {/* Section List */}
            <div className="space-y-1">
              {module1Sections.map((section) => {
                const Icon = section.icon
                const isActive = activeSection === section.id
                const isCompleted = completedSections.includes(section.id)

                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 p-3 text-left rounded-lg transition-colors ${
                      isActive ? "bg-blue-50 text-blue-700 border border-blue-200" : "hover:bg-gray-50 text-gray-700"
                    }`}
                    title={sidebarCollapsed ? section.title : undefined}
                  >
                    <Icon className="h-4 w-4 flex-shrink-0" />
                    {!sidebarCollapsed && (
                      <>
                        <span className="text-sm font-medium flex-1">{section.title}</span>
                        {isCompleted && <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />}
                      </>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 min-w-0">
          <div className="p-4 lg:p-8">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="outline" className="text-xs font-medium">
                    MODULE 1
                  </Badge>
                  <span className="text-sm text-gray-600 truncate">{currentSection?.title}</span>
                </div>
                <h1 className="text-2xl lg:text-3xl font-bold text-gray-900 truncate">
                  {currentSection?.title.replace(/^\d+\.\s*/, "")}
                </h1>
              </div>
              <Button
                onClick={markComplete}
                disabled={completedSections.includes(activeSection)}
                className="flex items-center gap-2 ml-4"
              >
                <CheckCircle className="h-4 w-4" />
                {completedSections.includes(activeSection) ? "Completed" : "Mark Complete"}
              </Button>
            </div>

            {/* Content */}
            <div className="max-w-full">
              <ContentSection activeSection={activeSection} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ContentSection({ activeSection }: { activeSection: string }) {
  const content = getContentForSection(activeSection)
  return <div className="prose max-w-none">{content}</div>
}

// Interactive Lab: Model Comparison Dashboard
function ModelComparisonDashboard() {
  const [selectedModels, setSelectedModels] = useState<string[]>(["linear-regression", "random-forest"])
  const [selectedMetric, setSelectedMetric] = useState("accuracy")

  const models = [
    { id: "linear-regression", name: "Linear Regression", accuracy: 0.85, precision: 0.82, recall: 0.88, f1: 0.85 },
    { id: "random-forest", name: "Random Forest", accuracy: 0.92, precision: 0.9, recall: 0.94, f1: 0.92 },
    { id: "svm", name: "Support Vector Machine", accuracy: 0.88, precision: 0.86, recall: 0.9, f1: 0.88 },
    { id: "neural-network", name: "Neural Network", accuracy: 0.94, precision: 0.93, recall: 0.95, f1: 0.94 },
  ]

  const metrics = [
    { id: "accuracy", name: "Accuracy", description: "Overall correctness of predictions" },
    { id: "precision", name: "Precision", description: "True positives / (True positives + False positives)" },
    { id: "recall", name: "Recall", description: "True positives / (True positives + False negatives)" },
    { id: "f1", name: "F1-Score", description: "Harmonic mean of precision and recall" },
  ]

  // Generate chart data points for visualization
  const generateChartData = () => {
    const thresholds = Array.from({ length: 21 }, (_, i) => i * 0.05)
    return thresholds.map((threshold) => ({
      threshold: threshold.toFixed(2),
      ...selectedModels.reduce(
        (acc, modelId) => {
          const model = models.find((m) => m.id === modelId)!
          const baseValue = model[selectedMetric as keyof typeof model] as number
          // Simulate how metrics change with threshold
          const variation = Math.sin(threshold * Math.PI) * 0.1
          acc[model.name] = Math.max(0.1, Math.min(0.99, baseValue + variation))
          return acc
        },
        {} as Record<string, number>,
      ),
    }))
  }

  const chartData = generateChartData()

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎯 Model Comparison Dashboard</h3>
        <p className="text-gray-700 mb-4">
          Compare different machine learning models side-by-side with various metrics and visualizations.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Interactive</Badge>
          <Badge variant="outline">Real-time</Badge>
          <Badge variant="outline">Comparative Analysis</Badge>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Model Selection */}
        <Card>
          <CardHeader>
            <CardTitle>Select Models</CardTitle>
            <CardDescription>Choose models to compare</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {models.map((model) => (
                <label key={model.id} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedModels([...selectedModels, model.id])
                      } else {
                        setSelectedModels(selectedModels.filter((id) => id !== model.id))
                      }
                    }}
                    className="rounded"
                  />
                  <span className="text-sm">{model.name}</span>
                </label>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Metric Selection */}
        <Card>
          <CardHeader>
            <CardTitle>Select Metric</CardTitle>
            <CardDescription>Choose evaluation metric</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {metrics.map((metric) => (
                <label key={metric.id} className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="metric"
                    value={metric.id}
                    checked={selectedMetric === metric.id}
                    onChange={(e) => setSelectedMetric(e.target.value)}
                    className="rounded"
                  />
                  <div>
                    <span className="text-sm font-medium">{metric.name}</span>
                    <p className="text-xs text-gray-500">{metric.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <Card>
          <CardHeader>
            <CardTitle>Comparison Results</CardTitle>
            <CardDescription>Model performance comparison</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {selectedModels.map((modelId) => {
                const model = models.find((m) => m.id === modelId)!
                const value = model[selectedMetric as keyof typeof model] as number
                return (
                  <div key={modelId} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>{model.name}</span>
                      <span className="font-medium">{(value * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={value * 100} className="h-2" />
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Interactive Chart Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Threshold vs Metrics Visualization</CardTitle>
          <CardDescription>How selected models perform across different thresholds</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full">
            <svg viewBox="0 0 800 300" className="w-full h-full">
              {/* Chart background */}
              <rect width="800" height="300" fill="#f9fafb" stroke="#e5e7eb" />

              {/* Grid lines */}
              {Array.from({ length: 11 }, (_, i) => (
                <g key={i}>
                  <line x1={80 + i * 64} y1="20" x2={80 + i * 64} y2="260" stroke="#e5e7eb" strokeWidth="1" />
                  <text x={80 + i * 64} y="280" textAnchor="middle" fontSize="10" fill="#6b7280">
                    {(i * 0.1).toFixed(1)}
                  </text>
                </g>
              ))}

              {Array.from({ length: 6 }, (_, i) => (
                <g key={i}>
                  <line x1="80" y1={20 + i * 48} x2="720" y2={20 + i * 48} stroke="#e5e7eb" strokeWidth="1" />
                  <text x="70" y={25 + i * 48} textAnchor="end" fontSize="10" fill="#6b7280">
                    {((1 - i * 0.2) * 100).toFixed(0)}%
                  </text>
                </g>
              ))}

              {/* Chart lines */}
              {selectedModels.map((modelId, index) => {
                const model = models.find((m) => m.id === modelId)!
                const color = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"][index]

                const points = chartData
                  .map((point, i) => {
                    const x = 80 + (i / (chartData.length - 1)) * 640
                    const y = 260 - (point[model.name] as number) * 240
                    return `${x},${y}`
                  })
                  .join(" ")

                return (
                  <g key={modelId}>
                    <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
                    {/* Legend */}
                    <rect x={550} y={30 + index * 25} width="15" height="3" fill={color} />
                    <text x={575} y={37 + index * 25} fontSize="12" fill="#374151">
                      {model.name}
                    </text>
                  </g>
                )
              })}

              {/* Axis labels */}
              <text x="400" y="295" textAnchor="middle" fontSize="12" fill="#374151">
                Threshold
              </text>
              <text x="25" y="150" textAnchor="middle" fontSize="12" fill="#374151" transform="rotate(-90 25 150)">
                {metrics.find((m) => m.id === selectedMetric)?.name} (%)
              </text>
            </svg>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Interactive Lab: Metric Sensitivity Explorer
function MetricSensitivityExplorer() {
  const [threshold, setThreshold] = useState(0.5)
  const [selectedMetric, setSelectedMetric] = useState("precision")

  const calculateMetrics = (threshold: number) => {
    // Simulated data based on threshold
    const precision = Math.max(0.1, Math.min(0.95, 0.8 - (threshold - 0.5) * 0.3))
    const recall = Math.max(0.1, Math.min(0.95, 0.7 + (threshold - 0.5) * 0.4))
    const f1 = (2 * precision * recall) / (precision + recall)
    const accuracy = Math.max(0.1, Math.min(0.95, 0.85 - Math.abs(threshold - 0.6) * 0.2))

    return { precision, recall, f1, accuracy }
  }

  const metrics = calculateMetrics(threshold)

  // Generate sensitivity curve data
  const generateSensitivityData = () => {
    const thresholds = Array.from({ length: 81 }, (_, i) => 0.1 + i * 0.01)
    return thresholds.map((t) => ({
      threshold: t,
      ...calculateMetrics(t),
    }))
  }

  const sensitivityData = generateSensitivityData()

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">⚡ Metric Sensitivity Explorer</h3>
        <p className="text-gray-700 mb-4">
          Explore how different metrics respond to changes in classification thresholds and model parameters.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Interactive</Badge>
          <Badge variant="outline">Real-time Updates</Badge>
          <Badge variant="outline">Sensitivity Analysis</Badge>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Threshold Control</CardTitle>
            <CardDescription>Adjust classification threshold</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Classification Threshold: {threshold.toFixed(2)}</label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(Number.parseFloat(e.target.value))}
                className="w-full mt-2"
              />
            </div>

            <div className="space-y-3">
              <h4 className="font-medium">Current Metrics</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded">
                  <div className="text-2xl font-bold text-blue-600">{(metrics.precision * 100).toFixed(1)}%</div>
                  <div className="text-sm text-blue-800">Precision</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded">
                  <div className="text-2xl font-bold text-green-600">{(metrics.recall * 100).toFixed(1)}%</div>
                  <div className="text-sm text-green-800">Recall</div>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded">
                  <div className="text-2xl font-bold text-purple-600">{(metrics.f1 * 100).toFixed(1)}%</div>
                  <div className="text-sm text-purple-800">F1-Score</div>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded">
                  <div className="text-2xl font-bold text-orange-600">{(metrics.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-sm text-orange-800">Accuracy</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Visualization */}
        <Card>
          <CardHeader>
            <CardTitle>Sensitivity Visualization</CardTitle>
            <CardDescription>How metrics change with threshold</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 w-full">
              <svg viewBox="0 0 400 200" className="w-full h-full">
                {/* Chart background */}
                <rect width="400" height="200" fill="#f9fafb" stroke="#e5e7eb" />

                {/* Grid lines */}
                {Array.from({ length: 6 }, (_, i) => (
                  <g key={i}>
                    <line x1={40 + i * 60} y1="10" x2={40 + i * 60} y2="170" stroke="#e5e7eb" strokeWidth="1" />
                    <text x={40 + i * 60} y="185" textAnchor="middle" fontSize="8" fill="#6b7280">
                      {(0.1 + i * 0.16).toFixed(1)}
                    </text>
                  </g>
                ))}

                {Array.from({ length: 6 }, (_, i) => (
                  <g key={i}>
                    <line x1="40" y1={10 + i * 32} x2="340" y2={10 + i * 32} stroke="#e5e7eb" strokeWidth="1" />
                    <text x="35" y={15 + i * 32} textAnchor="end" fontSize="8" fill="#6b7280">
                      {((1 - i * 0.2) * 100).toFixed(0)}%
                    </text>
                  </g>
                ))}

                {/* Metric curves */}
                {["precision", "recall", "f1", "accuracy"].map((metric, index) => {
                  const color = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"][index]

                  const points = sensitivityData
                    .map((point, i) => {
                      const x = 40 + ((point.threshold - 0.1) / 0.8) * 300
                      const y = 170 - (point[metric as keyof typeof point] as number) * 160
                      return `${x},${y}`
                    })
                    .join(" ")

                  return <polyline key={metric} points={points} fill="none" stroke={color} strokeWidth="2" />
                })}

                {/* Current threshold line */}
                <line
                  x1={40 + ((threshold - 0.1) / 0.8) * 300}
                  y1="10"
                  x2={40 + ((threshold - 0.1) / 0.8) * 300}
                  y2="170"
                  stroke="#ef4444"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />

                {/* Legend */}
                {["Precision", "Recall", "F1-Score", "Accuracy"].map((name, index) => (
                  <g key={name}>
                    <rect
                      x={350}
                      y={20 + index * 15}
                      width="10"
                      height="2"
                      fill={["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"][index]}
                    />
                    <text x={365} y={25 + index * 15} fontSize="8" fill="#374151">
                      {name}
                    </text>
                  </g>
                ))}
              </svg>
            </div>

            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span>Precision-Recall Trade-off</span>
                <span className="text-gray-500">Current: {threshold.toFixed(2)}</span>
              </div>
              <div className="text-xs text-gray-600">Lower thresholds increase recall but may decrease precision</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// Interactive Lab: Hyperparameter Optimization Tracker
function HyperparameterOptimizationTracker() {
  const [experiments, setExperiments] = useState([
    { id: 1, learningRate: 0.01, batchSize: 32, epochs: 100, accuracy: 0.85, status: "completed" as const },
    { id: 2, learningRate: 0.001, batchSize: 64, epochs: 150, accuracy: 0.88, status: "completed" as const },
    { id: 3, learningRate: 0.1, batchSize: 16, epochs: 80, accuracy: 0.82, status: "completed" as const },
    { id: 4, learningRate: 0.005, batchSize: 32, epochs: 120, accuracy: 0.91, status: "running" as const },
  ])

  const [newExperiment, setNewExperiment] = useState({
    learningRate: 0.01,
    batchSize: 32,
    epochs: 100,
  })

  const [isRunning, setIsRunning] = useState(false)

  const addExperiment = () => {
    if (isRunning) return

    setIsRunning(true)
    const experiment = {
      id: experiments.length + 1,
      ...newExperiment,
      accuracy: 0,
      status: "running" as const,
    }

    setExperiments([...experiments, experiment])

    // Simulate experiment running
    setTimeout(() => {
      const finalAccuracy = Math.random() * 0.3 + 0.7 // Random accuracy between 0.7-1.0
      setExperiments((prev) =>
        prev.map((exp) =>
          exp.id === experiment.id ? { ...exp, accuracy: finalAccuracy, status: "completed" as const } : exp,
        ),
      )
      setIsRunning(false)
    }, 3000)
  }

  const getBestExperiment = () => {
    const completedExperiments = experiments.filter((exp) => exp.status === "completed")
    if (completedExperiments.length === 0) return experiments[0]
    return completedExperiments.reduce((best, current) => (current.accuracy > best.accuracy ? current : best))
  }

  const bestExp = getBestExperiment()

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-50 to-teal-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎯 Hyperparameter Optimization Tracker</h3>
        <p className="text-gray-700 mb-4">
          Track and visualize hyperparameter optimization experiments to find the best model configuration. This tool
          helps you systematically explore different parameter combinations.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Experiment Tracking</Badge>
          <Badge variant="outline">Optimization</Badge>
          <Badge variant="outline">Parameter Tuning</Badge>
          <Badge variant="outline">Interactive</Badge>
        </div>
      </div>

      {/* Explanation Section */}
      <Card>
        <CardHeader>
          <CardTitle>How Hyperparameter Optimization Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Key Hyperparameters</h4>
              <ul className="text-sm space-y-2">
                <li>
                  <strong>Learning Rate:</strong> Controls how fast the model learns (0.001-0.1)
                </li>
                <li>
                  <strong>Batch Size:</strong> Number of samples processed together (16-128)
                </li>
                <li>
                  <strong>Epochs:</strong> Number of complete passes through the data (50-200)
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Optimization Strategy</h4>
              <ul className="text-sm space-y-2">
                <li>• Start with default values</li>
                <li>• Try different combinations systematically</li>
                <li>• Track performance for each experiment</li>
                <li>• Select best performing configuration</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* New Experiment */}
        <Card>
          <CardHeader>
            <CardTitle>New Experiment</CardTitle>
            <CardDescription>Configure hyperparameters for next experiment</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Learning Rate</label>
              <Input
                type="number"
                step="0.001"
                min="0.001"
                max="0.1"
                value={newExperiment.learningRate}
                onChange={(e) =>
                  setNewExperiment({ ...newExperiment, learningRate: Number.parseFloat(e.target.value) })
                }
                className="mt-1"
              />
              <p className="text-xs text-gray-500 mt-1">Typical range: 0.001 - 0.1</p>
            </div>
            <div>
              <label className="text-sm font-medium">Batch Size</label>
              <Input
                type="number"
                min="8"
                max="256"
                value={newExperiment.batchSize}
                onChange={(e) => setNewExperiment({ ...newExperiment, batchSize: Number.parseInt(e.target.value) })}
                className="mt-1"
              />
              <p className="text-xs text-gray-500 mt-1">Common values: 16, 32, 64, 128</p>
            </div>
            <div>
              <label className="text-sm font-medium">Epochs</label>
              <Input
                type="number"
                min="10"
                max="500"
                value={newExperiment.epochs}
                onChange={(e) => setNewExperiment({ ...newExperiment, epochs: Number.parseInt(e.target.value) })}
                className="mt-1"
              />
              <p className="text-xs text-gray-500 mt-1">Typical range: 50 - 200</p>
            </div>
            <Button onClick={addExperiment} className="w-full" disabled={isRunning}>
              {isRunning ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Experiment
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Experiment History */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Experiment History</CardTitle>
            <CardDescription>Track all optimization experiments and their results</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {experiments.map((exp) => (
                <div key={exp.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className="text-sm font-medium">#{exp.id}</div>
                    <div className="text-sm space-x-4">
                      <span>LR: {exp.learningRate}</span>
                      <span>BS: {exp.batchSize}</span>
                      <span>EP: {exp.epochs}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-sm font-medium">
                      {exp.status === "completed" ? (
                        `${(exp.accuracy * 100).toFixed(1)}%`
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
                          Running...
                        </div>
                      )}
                    </div>
                    <Badge variant={exp.status === "completed" ? "secondary" : "default"}>{exp.status}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Best Results */}
      <Card>
        <CardHeader>
          <CardTitle>Best Configuration Found</CardTitle>
          <CardDescription>Highest performing hyperparameters from all experiments</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{(bestExp.accuracy * 100).toFixed(1)}%</div>
              <div className="text-sm text-green-800">Best Accuracy</div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{bestExp.learningRate}</div>
              <div className="text-sm text-blue-800">Learning Rate</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{bestExp.batchSize}</div>
              <div className="text-sm text-purple-800">Batch Size</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{bestExp.epochs}</div>
              <div className="text-sm text-orange-800">Epochs</div>
            </div>
          </div>

          {bestExp.accuracy > 0 && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <h4 className="font-semibold text-blue-900 mb-2">Recommended Next Steps:</h4>
              <ul className="text-sm text-blue-800 space-y-1">
                {bestExp.accuracy > 0.9 ? (
                  <>
                    <li>• Excellent performance! Consider fine-tuning around these values</li>
                    <li>• Try learning rates ±20% from {bestExp.learningRate}</li>
                    <li>• Test with regularization techniques</li>
                  </>
                ) : bestExp.accuracy > 0.8 ? (
                  <>
                    <li>• Good performance. Try smaller learning rates for stability</li>
                    <li>• Experiment with different batch sizes</li>
                    <li>• Consider increasing epochs if not overfitting</li>
                  </>
                ) : (
                  <>
                    <li>• Performance needs improvement. Try different architectures</li>
                    <li>• Check data quality and preprocessing</li>
                    <li>• Consider feature engineering</li>
                  </>
                )}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Performance Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Experiment Performance Visualization</CardTitle>
          <CardDescription>Visual comparison of all completed experiments</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 w-full">
            <svg viewBox="0 0 800 200" className="w-full h-full">
              {/* Chart background */}
              <rect width="800" height="200" fill="#f9fafb" stroke="#e5e7eb" />

              {/* Grid lines */}
              {Array.from({ length: 6 }, (_, i) => (
                <g key={i}>
                  <line x1="60" y1={20 + i * 30} x2="740" y2={20 + i * 30} stroke="#e5e7eb" strokeWidth="1" />
                  <text x="50" y={25 + i * 30} textAnchor="end" fontSize="10" fill="#6b7280">
                    {((1 - i * 0.2) * 100).toFixed(0)}%
                  </text>
                </g>
              ))}

              {/* Experiment bars */}
              {experiments
                .filter((exp) => exp.status === "completed")
                .map((exp, index) => {
                  const x = 80 + index * 80
                  const height = exp.accuracy * 150
                  const y = 170 - height
                  const color = exp.id === bestExp.id ? "#10b981" : "#3b82f6"

                  return (
                    <g key={exp.id}>
                      <rect x={x} y={y} width="40" height={height} fill={color} opacity={0.8} />
                      <text x={x + 20} y="190" textAnchor="middle" fontSize="10" fill="#374151">
                        #{exp.id}
                      </text>
                      <text x={x + 20} y={y - 5} textAnchor="middle" fontSize="9" fill="#374151">
                        {(exp.accuracy * 100).toFixed(1)}%
                      </text>
                    </g>
                  )
                })}

              {/* Labels */}
              <text x="400" y="15" textAnchor="middle" fontSize="12" fill="#374151" fontWeight="bold">
                Experiment Performance Comparison
              </text>
            </svg>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Interactive Lab: Explainability Explorer
function ExplainabilityExplorer() {
  const [selectedFeature, setSelectedFeature] = useState("age")
  const [selectedInstance, setSelectedInstance] = useState(0)

  const features = [
    { name: "age", importance: 0.25, description: "Customer age" },
    { name: "income", importance: 0.35, description: "Annual income" },
    { name: "credit_score", importance: 0.2, description: "Credit score" },
    { name: "account_balance", importance: 0.15, description: "Account balance" },
    { name: "transaction_frequency", importance: 0.05, description: "Monthly transactions" },
  ]

  const instances = [
    {
      id: 0,
      age: 35,
      income: 75000,
      credit_score: 720,
      account_balance: 15000,
      transaction_frequency: 25,
      prediction: 0.85,
      actual: "approved",
      shapValues: { age: 0.15, income: 0.25, credit_score: 0.2, account_balance: 0.1, transaction_frequency: 0.15 },
    },
    {
      id: 1,
      age: 22,
      income: 35000,
      credit_score: 580,
      account_balance: 2000,
      transaction_frequency: 8,
      prediction: 0.25,
      actual: "denied",
      shapValues: { age: -0.2, income: -0.3, credit_score: -0.25, account_balance: -0.15, transaction_frequency: -0.1 },
    },
    {
      id: 2,
      age: 45,
      income: 120000,
      credit_score: 800,
      account_balance: 50000,
      transaction_frequency: 40,
      prediction: 0.95,
      actual: "approved",
      shapValues: { age: 0.2, income: 0.35, credit_score: 0.25, account_balance: 0.2, transaction_frequency: 0.15 },
    },
  ]

  const currentInstance = instances[selectedInstance]

  // Generate explanation text based on SHAP values
  const generateExplanation = () => {
    const sortedFeatures = Object.entries(currentInstance.shapValues)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 3)

    const explanations = sortedFeatures.map(([feature, value]) => {
      const impact = value > 0 ? "increases" : "decreases"
      const strength = Math.abs(value) > 0.2 ? "strongly" : "moderately"
      const featureDesc = features.find((f) => f.name === feature)?.description || feature
      return `${featureDesc} ${strength} ${impact} approval probability`
    })

    return explanations
  }

  const explanations = generateExplanation()

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🧠 Explainability Explorer</h3>
        <p className="text-gray-700 mb-4">
          Understand how your models make decisions with interactive explainability tools and visualizations.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Model Interpretability</Badge>
          <Badge variant="outline">Feature Importance</Badge>
          <Badge variant="outline">SHAP Values</Badge>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Feature Importance */}
        <Card>
          <CardHeader>
            <CardTitle>Global Feature Importance</CardTitle>
            <CardDescription>Overall feature contributions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {features.map((feature) => (
                <div key={feature.name} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{feature.description}</span>
                    <span>{(feature.importance * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={feature.importance * 100} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Instance Explanation */}
        <Card>
          <CardHeader>
            <CardTitle>Instance Explanation</CardTitle>
            <CardDescription>Why this prediction was made</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Select Instance</label>
                <select
                  value={selectedInstance}
                  onChange={(e) => setSelectedInstance(Number.parseInt(e.target.value))}
                  className="w-full mt-1 p-2 border rounded-md"
                >
                  {instances.map((instance) => (
                    <option key={instance.id} value={instance.id}>
                      Instance {instance.id + 1} - {instance.actual}
                    </option>
                  ))}
                </select>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Age:</span> {currentInstance.age}
                  </div>
                  <div>
                    <span className="font-medium">Income:</span> ${currentInstance.income.toLocaleString()}
                  </div>
                  <div>
                    <span className="font-medium">Credit Score:</span> {currentInstance.credit_score}
                  </div>
                  <div>
                    <span className="font-medium">Balance:</span> ${currentInstance.account_balance.toLocaleString()}
                  </div>
                  <div>
                    <span className="font-medium">Prediction:</span> {(currentInstance.prediction * 100).toFixed(1)}%
                  </div>
                  <div>
                    <span className="font-medium">Actual:</span> {currentInstance.actual}
                  </div>
                </div>
              </div>

              {/* Explanation Text */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">Why this prediction?</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  {explanations.map((explanation, index) => (
                    <li key={index}>• {explanation}</li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* SHAP Explanation Chart */}
      <Card>
        <CardHeader>
          <CardTitle>SHAP Values Visualization</CardTitle>
          <CardDescription>Feature contributions for selected instance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 w-full">
            <svg viewBox="0 0 600 200" className="w-full h-full">
              {/* Chart background */}
              <rect width="600" height="200" fill="#f9fafb" stroke="#e5e7eb" />

              {/* Center line */}
              <line x1="300" y1="20" x2="300" y2="180" stroke="#6b7280" strokeWidth="2" />

              {/* SHAP value bars */}
              {Object.entries(currentInstance.shapValues).map(([feature, value], index) => {
                const y = 30 + index * 30
                const barWidth = Math.abs(value) * 200
                const x = value > 0 ? 300 : 300 - barWidth
                const color = value > 0 ? "#10b981" : "#ef4444"

                return (
                  <g key={feature}>
                    {/* Bar */}
                    <rect x={x} y={y} width={barWidth} height={20} fill={color} opacity={0.7} />

                    {/* Feature label */}
                    <text x="10" y={y + 15} fontSize="12" fill="#374151">
                      {features.find((f) => f.name === feature)?.description}
                    </text>

                    {/* Value label */}
                    <text
                      x={value > 0 ? x + barWidth + 5 : x - 5}
                      y={y + 15}
                      fontSize="10"
                      fill="#6b7280"
                      textAnchor={value > 0 ? "start" : "end"}
                    >
                      {value.toFixed(2)}
                    </text>
                  </g>
                )
              })}

              {/* Labels */}
              <text x="150" y="15" textAnchor="middle" fontSize="12" fill="#ef4444">
                Decreases Probability
              </text>
              <text x="450" y="15" textAnchor="middle" fontSize="12" fill="#10b981">
                Increases Probability
              </text>

              {/* Base value indicator */}
              <text x="300" y="195" textAnchor="middle" fontSize="10" fill="#6b7280">
                Base Value: 0.5
              </text>
            </svg>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// ML Model Development Assignment
function MLModelDevelopmentAssignment() {
  const [activeTab, setActiveTab] = useState("overview")

  const assignments = [
    {
      id: 1,
      title: "Titanic Survival Prediction (Binary Classification)",
      dataset: "Titanic competition data with 891 passengers",
      algorithms: "Logistic Regression + Decision Tree (from scratch)",
      keySkills: "Binary classification, sigmoid function, gradient descent, entropy, information gain",
      features: "Missing value handling, feature engineering (titles, family size), categorical encoding",
      kaggleLink: "https://www.kaggle.com/c/titanic",
    },
    {
      id: 2,
      title: "House Price Prediction (Regression)",
      dataset: "House Prices Advanced Regression Techniques (1460 houses, 79 features)",
      algorithms: "Linear, Ridge, Lasso, Polynomial Regression (from scratch)",
      keySkills: "Normal equation, regularization, cross-validation, feature scaling",
      features: "Advanced feature engineering, multicollinearity handling, outlier detection",
      kaggleLink: "https://www.kaggle.com/c/house-prices-advanced-regression-techniques",
    },
  ]

  const hints = [
    "💡 For Titanic: Focus on feature engineering - extract titles from names, create family size features",
    "💡 For House Prices: Handle missing values carefully and consider log transformation for skewed target variable",
    "💡 Implement algorithms from scratch to understand the underlying mathematics",
    "💡 Use cross-validation to evaluate model performance and prevent overfitting",
    "💡 Feature scaling is crucial for gradient-based algorithms like logistic regression",
  ]

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          ML Model Development Assignment
        </CardTitle>
        <CardDescription>Complete these two machine learning assignments and submit your solutions</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="assignments">Assignments</TabsTrigger>
            <TabsTrigger value="resources">Resources</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-blue-900 mb-2">Assignment Overview</h3>
                <p className="text-blue-800">
                  Complete two comprehensive machine learning assignments covering both classification and regression.
                  Implement algorithms from scratch and work with real Kaggle datasets.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">📊 Datasets</h4>
                  <p className="text-green-800 text-sm">Real Kaggle competition datasets: Titanic and House Prices</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">⏱️ Duration</h4>
                  <p className="text-purple-800 text-sm">Estimated time: 6-8 hours total</p>
                </div>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-semibold text-yellow-900 mb-2">💡 Hints</h4>
                <ul className="space-y-2">
                  {hints.map((hint, index) => (
                    <li key={index} className="text-yellow-800 text-sm">
                      {hint}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="assignments" className="space-y-6">
            {assignments.map((assignment) => (
              <div key={assignment.id} className="border rounded-lg p-6">
                <h3 className="font-bold text-lg mb-4 text-blue-900">{assignment.title}</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div className="space-y-3">
                    <div>
                      <h4 className="font-semibold text-gray-700 mb-1">📊 Dataset</h4>
                      <p className="text-sm text-gray-600">{assignment.dataset}</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-700 mb-1">🤖 Algorithms</h4>
                      <p className="text-sm text-gray-600">{assignment.algorithms}</p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <h4 className="font-semibold text-gray-700 mb-1">🎯 Key Skills</h4>
                      <p className="text-sm text-gray-600">{assignment.keySkills}</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-700 mb-1">⚙️ Features</h4>
                      <p className="text-sm text-gray-600">{assignment.features}</p>
                    </div>
                  </div>
                </div>

                <Button variant="outline" className="w-full bg-transparent" asChild>
                  <a href={assignment.kaggleLink} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Access {assignment.title} Dataset on Kaggle
                  </a>
                </Button>
              </div>
            ))}

            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">📤 Submission</h3>
              <p className="text-blue-800 mb-3">
                Submit your implementations for both assignments using the link below:
              </p>
              <Button className="w-full" asChild>
                <a href="https://forms.google.com/ml-assignment-submission" target="_blank" rel="noopener noreferrer">
                  <Send className="h-4 w-4 mr-2" />
                  Submit Both Assignments
                </a>
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="resources" className="space-y-4">
            <div className="space-y-4">
              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2 flex items-center gap-2">
                  <ExternalLink className="h-4 w-4" />
                  Kaggle Datasets
                </h3>
                <div className="space-y-3">
                  <div>
                    <p className="font-medium text-gray-700">Titanic - Machine Learning from Disaster</p>
                    <Button variant="outline" size="sm" asChild>
                      <a href="https://www.kaggle.com/c/titanic" target="_blank" rel="noopener noreferrer">
                        Access Dataset
                      </a>
                    </Button>
                  </div>
                  <div>
                    <p className="font-medium text-gray-700">House Prices - Advanced Regression Techniques</p>
                    <Button variant="outline" size="sm" asChild>
                      <a
                        href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Access Dataset
                      </a>
                    </Button>
                  </div>
                </div>
              </div>

              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2">📚 Additional Resources</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Scikit-learn documentation for regression models</li>
                  <li>• Pandas documentation for data preprocessing</li>
                  <li>• Matplotlib/Seaborn for data visualization</li>
                  <li>• Cross-validation techniques for model evaluation</li>
                </ul>
              </div>

              <div className="border rounded-lg p-4">
                <h3 className="font-semibold mb-2">🎯 Learning Objectives</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Understand different evaluation metrics for regression</li>
                  <li>• Apply feature scaling techniques appropriately</li>
                  <li>• Implement basic regression models</li>
                  <li>• Evaluate model performance effectively</li>
                </ul>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

// Mini Project Component
function MiniProjectFraudDetection() {
  const [activePhase, setActivePhase] = useState("planning")
  const [completedPhases, setCompletedPhases] = useState<string[]>([])

  const phases = [
    {
      id: "planning",
      title: "Project Planning",
      description: "Define requirements and architecture",
      icon: Target,
      tasks: [
        "Define fraud detection requirements",
        "Design system architecture",
        "Plan data pipeline",
        "Set up project repository",
      ],
    },
    {
      id: "data",
      title: "Data Preparation",
      description: "Collect and prepare fraud detection dataset",
      icon: Database,
      tasks: [
        "Load credit card transaction data",
        "Perform exploratory data analysis",
        "Handle class imbalance",
        "Feature engineering",
      ],
    },
    {
      id: "modeling",
      title: "Model Development",
      description: "Build and train fraud detection models",
      icon: Brain,
      tasks: [
        "Train multiple ML models",
        "Evaluate model performance",
        "Handle false positives/negatives",
        "Select best performing model",
      ],
    },
    {
      id: "deployment",
      title: "Model Deployment",
      description: "Deploy model to production environment",
      icon: Rocket,
      tasks: ["Containerize the model", "Set up API endpoints", "Deploy to cloud platform", "Configure load balancing"],
    },
    {
      id: "monitoring",
      title: "Monitoring & Maintenance",
      description: "Monitor model performance and maintain system",
      icon: BarChart3,
      tasks: [
        "Set up performance monitoring",
        "Configure alerting system",
        "Implement model retraining",
        "Create maintenance procedures",
      ],
    },
  ]

  const markPhaseComplete = (phaseId: string) => {
    if (!completedPhases.includes(phaseId)) {
      setCompletedPhases([...completedPhases, phaseId])
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-red-50 to-orange-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🛡️ Mini-Project: Deploying a Fraud Detection Model</h3>
        <p className="text-gray-700 mb-4">
          Complete end-to-end project deploying a fraud detection model from development to production with monitoring
          and maintenance. This project uses real credit card transaction data to detect fraudulent activities.
        </p>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">End-to-End Project</Badge>
          <Badge variant="outline">Real-world Application</Badge>
          <Badge variant="outline">Production Deployment</Badge>
          <Badge variant="outline">Fraud Detection</Badge>
        </div>
      </div>

      {/* Dataset Information */}
      <Card>
        <CardHeader>
          <CardTitle>📊 Dataset Information</CardTitle>
          <CardDescription>Credit Card Fraud Detection Dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Dataset Details</h4>
              <ul className="text-sm space-y-2">
                <li>
                  <strong>Source:</strong> European cardholders transactions (2013)
                </li>
                <li>
                  <strong>Size:</strong> 284,807 transactions
                </li>
                <li>
                  <strong>Features:</strong> 30 (28 PCA + Time + Amount)
                </li>
                <li>
                  <strong>Fraud Cases:</strong> 492 (0.172% of all transactions)
                </li>
                <li>
                  <strong>File Size:</strong> ~150 MB
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Download Links</h4>
              <div className="space-y-2">
                <a
                  href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors"
                >
                  <div className="font-medium text-blue-900">Kaggle Dataset ↗</div>
                  <div className="text-sm text-blue-700">Primary source with full documentation</div>
                </a>
                <a
                  href="https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors"
                >
                  <div className="font-medium text-green-900">GitHub Repository ↗</div>
                  <div className="text-sm text-green-700">Code examples and preprocessing scripts</div>
                </a>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Possible Outcomes */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 Possible Model Outcomes & Interpretations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <h4 className="font-semibold text-green-800 mb-2">Excellent Performance</h4>
              <ul className="text-sm text-green-700 space-y-1">
                <li>
                  <strong>Precision:</strong> &gt; 0.9
                </li>
                <li>
                  <strong>Recall:</strong> &gt; 0.8
                </li>
                <li>
                  <strong>F1-Score:</strong> &gt; 0.85
                </li>
                <li>
                  <strong>AUC-ROC:</strong> &gt; 0.95
                </li>
              </ul>
              <p className="text-xs text-green-600 mt-2">
                <strong>Next Steps:</strong> Deploy to production, set up monitoring
              </p>
            </div>

            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <h4 className="font-semibold text-yellow-800 mb-2">Good Performance</h4>
              <ul className="text-sm text-yellow-700 space-y-1">
                <li>
                  <strong>Precision:</strong> 0.7 - 0.9
                </li>
                <li>
                  <strong>Recall:</strong> 0.6 - 0.8
                </li>
                <li>
                  <strong>F1-Score:</strong> 0.65 - 0.85
                </li>
                <li>
                  <strong>AUC-ROC:</strong> 0.85 - 0.95
                </li>
              </ul>
              <p className="text-xs text-yellow-600 mt-2">
                <strong>Next Steps:</strong> Feature engineering, hyperparameter tuning
              </p>
            </div>

            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <h4 className="font-semibold text-red-800 mb-2">Needs Improvement</h4>
              <ul className="text-sm text-red-700 space-y-1">
                <li>
                  <strong>Precision:</strong> &lt; 0.7
                </li>
                <li>
                  <strong>Recall:</strong> &lt; 0.6
                </li>
                <li>
                  <strong>F1-Score:</strong> &lt; 0.65
                </li>
                <li>
                  <strong>AUC-ROC:</strong> &lt; 0.85
                </li>
              </ul>
              <p className="text-xs text-red-600 mt-2">
                <strong>Next Steps:</strong> Try different algorithms, more data preprocessing
              </p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-blue-900 mb-2">Understanding Fraud Detection Metrics</h4>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
              <div>
                <p>
                  <strong>Precision:</strong> Of all transactions flagged as fraud, how many were actually fraud?
                </p>
                <p>
                  <strong>Recall:</strong> Of all actual fraud cases, how many did we catch?
                </p>
              </div>
              <div>
                <p>
                  <strong>False Positives:</strong> Legitimate transactions flagged as fraud (customer inconvenience)
                </p>
                <p>
                  <strong>False Negatives:</strong> Fraud cases missed by the model (financial loss)
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Usage Examples */}
      <Card>
        <CardHeader>
          <CardTitle>🔧 Using the Trained Model</CardTitle>
          <CardDescription>Examples of making predictions with new transaction data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold mb-3">Example 1: Single Transaction Prediction</h4>
              <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                <pre className="text-sm">
                  {`import joblib
import numpy as np

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# New transaction data (example values)
new_transaction = np.array([[
    -1.359807,  # V1 (PCA component)
    -0.072781,  # V2
    2.536347,   # V3
    1.378155,   # V4
    # ... (other V features)
    149.62,     # Amount
    0           # Time (seconds from first transaction)
]])

# Scale the features
new_transaction_scaled = scaler.transform(new_transaction)

# Make prediction
prediction = model.predict(new_transaction_scaled)
probability = model.predict_proba(new_transaction_scaled)

print(f"Prediction: {'FRAUD' if prediction[0] == 1 else 'LEGITIMATE'}")
print(f"Fraud Probability: {probability[0][1]:.4f}")

# Output:
# Prediction: LEGITIMATE
# Fraud Probability: 0.0234`}
                </pre>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Example 2: Batch Processing</h4>
              <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                <pre className="text-sm">
                  {`import pandas as pd

# Load batch of new transactions
new_transactions = pd.read_csv('new_transactions.csv')

# Preprocess (same as training data)
features = new_transactions.drop(['Time'], axis=1)  # Remove non-predictive features
features_scaled = scaler.transform(features)

# Make predictions
predictions = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)[:, 1]

# Add results to dataframe
new_transactions['fraud_prediction'] = predictions
new_transactions['fraud_probability'] = probabilities

# Flag high-risk transactions
high_risk = new_transactions[new_transactions['fraud_probability'] > 0.5]
print(f"Found {len(high_risk)} high-risk transactions")

# Save results
new_transactions.to_csv('transactions_with_predictions.csv', index=False)`}
                </pre>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Example 3: Real-time API Usage</h4>
              <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                <pre className="text-sm">
                  {`import requests
import json

# API endpoint
url = "http://your-fraud-detection-api.com/predict"

# Transaction data
transaction_data = {
    "amount": 149.62,
    "v1": -1.359807,
    "v2": -0.072781,
    "v3": 2.536347,
    # ... other features
}

# Make API request
response = requests.post(url, json=transaction_data)
result = response.json()

print(f"Transaction ID: {result['transaction_id']}")
print(f"Fraud Risk: {result['fraud_probability']:.4f}")
print(f"Decision: {result['decision']}")

if result['fraud_probability'] > 0.7:
    print("⚠️  HIGH RISK - Block transaction and notify customer")
elif result['fraud_probability'] > 0.3:
    print("⚡ MEDIUM RISK - Require additional verification")
else:
    print("✅ LOW RISK - Approve transaction")`}
                </pre>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Project Phases */}
      <Card>
        <CardHeader>
          <CardTitle>Project Phases</CardTitle>
          <CardDescription>
            {completedPhases.length}/{phases.length} completed
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {phases.map((phase, index) => {
              const Icon = phase.icon
              const isActive = activePhase === phase.id
              const isCompleted = completedPhases.includes(phase.id)

              return (
                <button
                  key={phase.id}
                  onClick={() => setActivePhase(phase.id)}
                  className={`w-full flex items-center gap-4 p-4 text-left rounded-lg border transition-colors ${
                    isActive ? "bg-blue-50 border-blue-200" : "hover:bg-gray-50"
                  }`}
                >
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gray-100">
                    <span className="text-sm font-medium">{String(index + 1).padStart(2, "0")}</span>
                    {isCompleted ? (
                      <CheckCircle className="h-4 w-4 text-green-500 ml-2" />
                    ) : (
                      <Icon className="h-4 w-4 text-gray-500 ml-2" />
                    )}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold">{phase.title}</h4>
                    <p className="text-sm text-gray-600">{phase.description}</p>
                  </div>
                </button>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Phase Details */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>{phases.find((p) => p.id === activePhase)?.title}</CardTitle>
              <CardDescription>{phases.find((p) => p.id === activePhase)?.description}</CardDescription>
            </div>
            <Button onClick={() => markPhaseComplete(activePhase)} disabled={completedPhases.includes(activePhase)}>
              {completedPhases.includes(activePhase) ? (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Done
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Complete
                </>
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold mb-3">Phase Tasks</h4>
              <ul className="space-y-2">
                {phases
                  .find((p) => p.id === activePhase)
                  ?.tasks.map((task, index) => (
                    <li key={index} className="flex items-center gap-3">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
                        {index + 1}
                      </span>
                      <span className="text-sm">{task}</span>
                    </li>
                  ))}
              </ul>
            </div>

            {activePhase === "planning" && (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Project Overview</h4>
                  <p className="text-gray-600 text-sm">
                    Build a real-time fraud detection system that can process credit card transactions and identify
                    potentially fraudulent activities with high accuracy while minimizing false positives.
                  </p>
                </div>
              </div>
            )}

            {activePhase === "data" && (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Dataset Loading & Exploration</h4>
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      {`# Load and explore fraud detection dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Normal cases: {len(df) - df['Class'].sum()}")

# Check class distribution
fraud_ratio = df['Class'].mean()
print(f"Fraud ratio: {fraud_ratio:.4f}")

# Visualize class imbalance
plt.figure(figsize=(8, 6))
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')
plt.show()

# Check for missing values
print("Missing values:")
print(df.isnull().sum().sum())

# Basic statistics
print("Amount statistics:")
print(df['Amount'].describe())`}
                    </pre>
                  </div>
                </div>
              </div>
            )}

            {activePhase === "modeling" && (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Model Training & Evaluation</h4>
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      {`# Train fraud detection models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Handle class imbalance with SMOTE
from imblearn.over_sampling import SMOTE

# Prepare data
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Evaluate performance
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save model
import joblib
joblib.dump(rf_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')`}
                    </pre>
                  </div>
                </div>
              </div>
            )}

            {activePhase === "deployment" && (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Deployment Configuration</h4>
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      {`# Docker configuration for model deployment
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# FastAPI application (main.py)
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model at startup
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

class Transaction(BaseModel):
    features: list

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    # Preprocess
    features = np.array(transaction.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability),
        "decision": "BLOCK" if probability > 0.5 else "APPROVE"
    }`}
                    </pre>
                  </div>
                </div>
              </div>
            )}

            {activePhase === "monitoring" && (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Monitoring & Maintenance Setup</h4>
                  <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                    <pre className="text-sm">
                      {`# Model monitoring script
import pandas as pd
import numpy as np
from datetime import datetime
import logging

def monitor_model_performance():
    # Load recent predictions
    recent_data = pd.read_csv('recent_predictions.csv')
    
    # Calculate metrics
    false_positive_rate = calculate_fpr(recent_data)
    response_time = recent_data['response_time'].mean()
    
    # Check thresholds
    if false_positive_rate > 0.05:  # 5% threshold
        send_alert("High false positive rate detected")
    
    if response_time > 100:  # 100ms threshold
        send_alert("High response time detected")
    
    # Log metrics
    logging.info(f"FPR: {false_positive_rate:.4f}, RT: {response_time:.2f}ms")

# Run monitoring every hour
if __name__ == "__main__":
    monitor_model_performance()`}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function getContentForSection(sectionId: string) {
  const contentMap: Record<string, React.ReactNode> = {
    "problem-definition": (
      <div className="space-y-6">
        <div className="prose max-w-none">
          <p>
            Every successful machine learning (ML) project starts with a clear understanding of the problem you're
            trying to solve. In this initial phase, the goal is to define the business problem in a way that can be
            addressed with data and algorithms.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Overview</CardTitle>
          </CardHeader>
          <CardContent>
            A well-defined problem statement is crucial. It guides data collection, model selection, and evaluation.
            Without a clear problem, an ML project can quickly become unfocused and fail to deliver business value.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Example: Customer Churn</CardTitle>
          </CardHeader>
          <CardContent>
            A company might want to reduce customer churn. The ML problem here would be: "Can we predict which customers
            are likely to leave in the next 30 days?"
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>SMART Framework</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Specific:</div>
                <div>Clearly defined problem statement.</div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Measurable:</div>
                <div>Define metrics for success (e.g., churn prediction accuracy).</div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Achievable:</div>
                <div>Ensure data and resources are available.</div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Relevant:</div>
                <div>Align with business objectives.</div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Time-bound:</div>
                <div>Define a timeframe for prediction (e.g., next 30 days).</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    ),

    "data-collection": (
      <div className="space-y-6">
        <div className="prose max-w-none">
          <p>
            Data is the foundation of any machine learning model. Once the problem is clearly defined, the next step is
            to gather data relevant to solving it.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>The Foundation of ML</CardTitle>
          </CardHeader>
          <CardContent>
            Quality data is more important than sophisticated algorithms. The saying "garbage in, garbage out"
            particularly applies to machine learning projects.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Data Sources</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">Structured Data</h4>
                <ul className="space-y-1 text-sm">
                  <li>• CSV files</li>
                  <li>• Databases (SQL)</li>
                  <li>• Spreadsheets</li>
                  <li>• APIs</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Unstructured Data</h4>
                <ul className="space-y-1 text-sm">
                  <li>• Images</li>
                  <li>• Videos</li>
                  <li>• Text documents</li>
                  <li>• Audio files</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Housing Price Example</CardTitle>
          </CardHeader>
          <CardContent>For housing price prediction, structured data might include:</CardContent>
          <CardContent>
            <ul className="space-y-1 text-sm">
              <li>• Number of rooms</li>
              <li>• Location coordinates</li>
              <li>• Area size (sq ft)</li>
              <li>• Year built</li>
              <li>• Neighborhood crime rates</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    ),

    "model-training": (
      <div className="space-y-6">
        <div className="prose max-w-none">
          <p>
            Model training is where the magic happens - algorithms learn patterns from your data to make predictions or
            classifications on new, unseen data.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Training Process</CardTitle>
          </CardHeader>
          <CardContent>
            During training, the model adjusts its internal parameters to minimize prediction errors on the training
            data. The goal is to find patterns that generalize well to new data.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Evaluation Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <h4 className="font-semibold mb-2">Classification</h4>
                <ul className="space-y-1 text-sm">
                  <li>• Accuracy</li>
                  <li>• Precision</li>
                  <li>• Recall</li>
                  <li>• F1-Score</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Regression</h4>
                <ul className="space-y-1 text-sm">
                  <li>• MSE</li>
                  <li>• RMSE</li>
                  <li>• MAE</li>
                  <li>• R²</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Cross-Validation</h4>
                <ul className="space-y-1 text-sm">
                  <li>• K-Fold CV</li>
                  <li>• Stratified CV</li>
                  <li>• Time Series CV</li>
                  <li>• Leave-One-Out</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    ),

    "model-deployment": (
      <div className="space-y-6">
        <div className="prose max-w-none">
          <p>
            Once a model performs well in testing, it needs to be deployed into a production environment where it can
            start generating predictions in real time or batch mode.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Deployment Strategies</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Real-time Inference</div>
                <div>Deploy models as APIs for immediate predictions</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Batch Processing</div>
                <div>Process large datasets periodically</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="font-semibold text-blue-600 min-w-0">Edge Deployment</div>
                <div>Deploy models on edge devices for low-latency inference</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Monitoring</CardTitle>
          </CardHeader>
          <CardContent>
            Continuous monitoring ensures your model maintains performance over time and alerts you to potential issues
            like data drift or model degradation.
          </CardContent>
        </Card>
      </div>
    ),

    iteration: (
      <div className="space-y-6">
        <div className="prose max-w-none">
          <p>
            Machine learning is an iterative process. Based on model performance and feedback, you'll need to refine
            your approach, retrain models, and continuously improve results.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Continuous Improvement</CardTitle>
          </CardHeader>
          <CardContent>
            The ML lifecycle doesn't end with deployment. Regular model updates, feature engineering, and performance
            monitoring are essential for maintaining model effectiveness.
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Feedback Loop</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
                  1
                </div>
                <div>Monitor model performance in production</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
                  2
                </div>
                <div>Collect new data and feedback</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
                  3
                </div>
                <div>Identify areas for improvement</div>
              </div>
              <div className="flex items-start gap-3">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
                  4
                </div>
                <div>Retrain and redeploy improved models</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    ),

    "interactive-lab-comparison": <ModelComparisonDashboard />,
    "interactive-lab-sensitivity": <MetricSensitivityExplorer />,
    "interactive-lab-optimization": <HyperparameterOptimizationTracker />,
    "interactive-lab-explainability": <ExplainabilityExplorer />,
    "coding-assignment": <MLModelDevelopmentAssignment />,
    "mini-project": <MiniProjectFraudDetection />,
  }

  return (
    contentMap[sectionId] || (
      <div className="text-center py-8 text-gray-500">Content for this section is being developed.</div>
    )
  )
}
