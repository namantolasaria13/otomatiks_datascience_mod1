"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Terminal, BarChart, Play, Code, Database, Brain, Target } from "lucide-react"

const labTools = [
  {
    id: "ml-lifecycle",
    title: "🎯 ML Lifecycle Explorer",
    description: "Interactive flowchart of the complete ML pipeline",
    icon: Target,
    category: "Overview",
  },
  {
    id: "data-detective",
    title: "🔍 Data Detective",
    description: "Drag-and-drop data source matching game",
    icon: Database,
    category: "Data",
  },
  {
    id: "dvc-playground",
    title: "⚡ DVC Playground",
    description: "Terminal simulator for DVC commands",
    icon: Terminal,
    category: "Versioning",
  },
  {
    id: "model-lab",
    title: "🧪 Model Laboratory",
    description: "Interactive model comparison and tuning",
    icon: Brain,
    category: "Modeling",
  },
  {
    id: "metrics-dashboard",
    title: "📊 Metrics Dashboard",
    description: "Real-time model performance visualization",
    icon: BarChart,
    category: "Evaluation",
  },
  {
    id: "code-sandbox",
    title: "💻 Live Code Editor",
    description: "Write and execute ML code in the browser",
    icon: Code,
    category: "Practice",
  },
]

export function InteractiveLabSection() {
  const [activeTool, setActiveTool] = useState("ml-lifecycle")

  return (
    <div className="space-y-6">
      {/* Lab Overview */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg border">
        <h3 className="text-2xl font-bold mb-4">🎛️ Interactive ML Laboratory</h3>
        <p className="text-gray-700 mb-4">
          Hands-on interactive tools and dashboards to explore machine learning concepts. Each tool provides a unique
          way to understand different aspects of the ML workflow.
        </p>

        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Interactive</Badge>
          <Badge variant="outline">Visual Learning</Badge>
          <Badge variant="outline">Hands-on</Badge>
          <Badge variant="outline">Real-time</Badge>
        </div>
      </div>

      {/* Lab Tools Grid */}
      <div className="grid lg:grid-cols-4 gap-6">
        {/* Tools Navigation */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Lab Tools</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-2 p-4">
                {labTools.map((tool) => {
                  const Icon = tool.icon
                  const isActive = activeTool === tool.id

                  return (
                    <Button
                      key={tool.id}
                      variant={isActive ? "default" : "ghost"}
                      className="w-full justify-start text-left h-auto p-3"
                      onClick={() => setActiveTool(tool.id)}
                    >
                      <div className="flex items-start gap-3">
                        <Icon className="h-5 w-5 mt-0.5" />
                        <div className="flex-1">
                          <div className="font-medium text-sm">{tool.title}</div>
                          <p className="text-xs text-gray-600 mt-1">{tool.description}</p>
                          <Badge variant="secondary" className="text-xs mt-2">
                            {tool.category}
                          </Badge>
                        </div>
                      </div>
                    </Button>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tool Content */}
        <div className="lg:col-span-3">
          <LabToolContent toolId={activeTool} />
        </div>
      </div>
    </div>
  )
}

function LabToolContent({ toolId }: { toolId: string }) {
  const toolContent = getLabToolContent(toolId)

  return (
    <Card className="min-h-[600px]">
      <CardHeader>
        <CardTitle className="text-xl">{toolContent.title}</CardTitle>
        <CardDescription className="text-base">{toolContent.description}</CardDescription>
      </CardHeader>
      <CardContent>{toolContent.content}</CardContent>
    </Card>
  )
}

function getLabToolContent(toolId: string) {
  const contentMap: Record<string, any> = {
    "ml-lifecycle": {
      title: "🎯 ML Lifecycle Visual Explorer",
      description: "Interactive flowchart showing each stage of the machine learning pipeline",
      content: <MLLifecycleExplorer />,
    },
    "data-detective": {
      title: "🔍 Data Detective",
      description: "Match data sources to their best use cases",
      content: <DataDetectiveGame />,
    },
    "dvc-playground": {
      title: "⚡ DVC Playground",
      description: "Practice DVC commands in a simulated terminal environment",
      content: <DVCPlayground />,
    },
    "model-lab": {
      title: "🧪 Model Laboratory",
      description: "Compare different models and tune hyperparameters",
      content: <ModelLaboratory />,
    },
    "metrics-dashboard": {
      title: "📊 Metrics Dashboard",
      description: "Visualize model performance metrics in real-time",
      content: <MetricsDashboard />,
    },
    "code-sandbox": {
      title: "💻 Live Code Editor",
      description: "Write and execute machine learning code",
      content: <CodeSandbox />,
    },
  }

  return (
    contentMap[toolId] || {
      title: "Tool Not Found",
      description: "This tool is under development",
      content: <div>Content coming soon...</div>,
    }
  )
}

// Individual Lab Tool Components
function MLLifecycleExplorer() {
  const [activeStage, setActiveStage] = useState<string | null>(null)

  const stages = [
    {
      id: "problem",
      title: "Problem Definition",
      color: "bg-blue-500",
      description: "Define the business problem and success metrics",
    },
    { id: "data", title: "Data Collection", color: "bg-green-500", description: "Gather and prepare relevant data" },
    {
      id: "explore",
      title: "Data Exploration",
      color: "bg-yellow-500",
      description: "Analyze and understand the data",
    },
    { id: "model", title: "Model Training", color: "bg-purple-500", description: "Build and train ML models" },
    { id: "evaluate", title: "Model Evaluation", color: "bg-red-500", description: "Assess model performance" },
    { id: "deploy", title: "Deployment", color: "bg-indigo-500", description: "Deploy model to production" },
  ]

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h4 className="font-semibold mb-2">Click on any stage to learn more</h4>
        <p className="text-sm text-gray-600">Interactive ML Pipeline Visualization</p>
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        {stages.map((stage, index) => (
          <div key={stage.id} className="flex items-center">
            <Button
              variant={activeStage === stage.id ? "default" : "outline"}
              className="h-16 w-32 flex-col"
              onClick={() => setActiveStage(activeStage === stage.id ? null : stage.id)}
            >
              <div className={`w-4 h-4 rounded-full ${stage.color} mb-1`}></div>
              <span className="text-xs">{stage.title}</span>
            </Button>
            {index < stages.length - 1 && <div className="w-8 h-0.5 bg-gray-300 mx-2"></div>}
          </div>
        ))}
      </div>

      {activeStage && (
        <Card className="mt-6">
          <CardContent className="p-6">
            <h5 className="font-semibold mb-2">{stages.find((s) => s.id === activeStage)?.title}</h5>
            <p className="text-gray-700">{stages.find((s) => s.id === activeStage)?.description}</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function DataDetectiveGame() {
  const [score, setScore] = useState(0)
  const [matches, setMatches] = useState<Record<string, string>>({})

  const dataSources = [
    { id: "csv", name: "CSV Files", icon: "📄" },
    { id: "api", name: "REST APIs", icon: "🔗" },
    { id: "sql", name: "SQL Database", icon: "🗄️" },
    { id: "images", name: "Image Files", icon: "🖼️" },
  ]

  const useCases = [
    { id: "tabular", name: "Tabular Analysis", correct: "csv" },
    { id: "realtime", name: "Real-time Data", correct: "api" },
    { id: "structured", name: "Structured Queries", correct: "sql" },
    { id: "vision", name: "Computer Vision", correct: "images" },
  ]

  const handleMatch = (sourceId: string, useCaseId: string) => {
    const useCase = useCases.find((uc) => uc.id === useCaseId)
    if (useCase && useCase.correct === sourceId) {
      setScore(score + 1)
      setMatches({ ...matches, [useCaseId]: sourceId })
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h4 className="font-semibold mb-2">Match Data Sources to Use Cases</h4>
        <Badge variant="secondary">Score: {score}/4</Badge>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <h5 className="font-semibold mb-4">Data Sources</h5>
          <div className="space-y-2">
            {dataSources.map((source) => (
              <Card key={source.id} className="p-4 cursor-pointer hover:bg-gray-50">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{source.icon}</span>
                  <span className="font-medium">{source.name}</span>
                </div>
              </Card>
            ))}
          </div>
        </div>

        <div>
          <h5 className="font-semibold mb-4">Use Cases</h5>
          <div className="space-y-2">
            {useCases.map((useCase) => (
              <Card
                key={useCase.id}
                className={`p-4 cursor-pointer hover:bg-gray-50 ${
                  matches[useCase.id] ? "bg-green-50 border-green-200" : ""
                }`}
                onClick={() => {
                  // Simplified matching - in real implementation, this would be drag-and-drop
                  const correctSource = dataSources.find((s) => s.id === useCase.correct)
                  if (correctSource) {
                    handleMatch(correctSource.id, useCase.id)
                  }
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{useCase.name}</span>
                  {matches[useCase.id] && <span className="text-green-500">✓</span>}
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {score === 4 && (
        <div className="text-center p-4 bg-green-50 rounded-lg">
          <h5 className="font-semibold text-green-800">🎉 Perfect Score!</h5>
          <p className="text-green-700">You've mastered data source selection!</p>
        </div>
      )}
    </div>
  )
}

function DVCPlayground() {
  const [commands, setCommands] = useState<string[]>([])
  const [currentCommand, setCurrentCommand] = useState("")

  const dvcCommands = [
    { cmd: "dvc init", output: "Initialized DVC repository" },
    { cmd: "dvc add data.csv", output: "Added data.csv to DVC tracking" },
    { cmd: "dvc push", output: "Pushed data to remote storage" },
    { cmd: "dvc status", output: "Data pipeline is up to date" },
  ]

  const executeCommand = () => {
    const matchedCommand = dvcCommands.find((dc) => dc.cmd === currentCommand.trim())
    if (matchedCommand) {
      setCommands([...commands, `$ ${currentCommand}`, matchedCommand.output])
    } else {
      setCommands([
        ...commands,
        `$ ${currentCommand}`,
        "Command not recognized. Try: dvc init, dvc add, dvc push, dvc status",
      ])
    }
    setCurrentCommand("")
  }

  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold mb-2">DVC Command Simulator</h4>
        <p className="text-sm text-gray-600">Try these commands: dvc init, dvc add data.csv, dvc push, dvc status</p>
      </div>

      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm min-h-[300px]">
        <div className="mb-4">
          {commands.map((cmd, index) => (
            <div key={index} className={cmd.startsWith("$") ? "text-yellow-400" : "text-green-400"}>
              {cmd}
            </div>
          ))}
        </div>

        <div className="flex items-center">
          <span className="text-yellow-400 mr-2">$</span>
          <input
            type="text"
            value={currentCommand}
            onChange={(e) => setCurrentCommand(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && executeCommand()}
            className="bg-transparent border-none outline-none flex-1 text-green-400"
            placeholder="Enter DVC command..."
          />
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Available Commands</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm space-y-1">
              {dvcCommands.map((cmd, index) => (
                <li key={index} className="font-mono text-gray-700">
                  {cmd.cmd}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">DVC Workflow</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="text-sm space-y-1 list-decimal list-inside">
              <li>Initialize DVC in your project</li>
              <li>Add datasets to version control</li>
              <li>Push data to remote storage</li>
              <li>Check pipeline status</li>
            </ol>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function ModelLaboratory() {
  const [selectedModel, setSelectedModel] = useState("linear")
  const [hyperparams, setHyperparams] = useState({ alpha: 1.0, n_estimators: 100 })

  const models = [
    { id: "linear", name: "Linear Regression", accuracy: 0.75 },
    { id: "forest", name: "Random Forest", accuracy: 0.85 },
    { id: "svm", name: "Support Vector Machine", accuracy: 0.78 },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold mb-2">Model Comparison Lab</h4>
        <p className="text-sm text-gray-600">Compare different models and tune hyperparameters</p>
      </div>

      <Tabs value={selectedModel} onValueChange={setSelectedModel}>
        <TabsList className="grid w-full grid-cols-3">
          {models.map((model) => (
            <TabsTrigger key={model.id} value={model.id}>
              {model.name}
            </TabsTrigger>
          ))}
        </TabsList>

        {models.map((model) => (
          <TabsContent key={model.id} value={model.id} className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>{model.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Accuracy Score</label>
                    <Progress value={model.accuracy * 100} className="mt-2" />
                    <span className="text-sm text-gray-600">{(model.accuracy * 100).toFixed(1)}%</span>
                  </div>

                  {model.id === "forest" && (
                    <div>
                      <label className="text-sm font-medium">Number of Estimators</label>
                      <input
                        type="range"
                        min="50"
                        max="200"
                        value={hyperparams.n_estimators}
                        onChange={(e) =>
                          setHyperparams({ ...hyperparams, n_estimators: Number.parseInt(e.target.value) })
                        }
                        className="w-full mt-2"
                      />
                      <span className="text-sm text-gray-600">{hyperparams.n_estimators}</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}

function MetricsDashboard() {
  const metrics = [
    { name: "Accuracy", value: 0.85, color: "bg-green-500" },
    { name: "Precision", value: 0.82, color: "bg-blue-500" },
    { name: "Recall", value: 0.78, color: "bg-purple-500" },
    { name: "F1-Score", value: 0.8, color: "bg-orange-500" },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold mb-2">Real-time Model Metrics</h4>
        <p className="text-sm text-gray-600">Monitor your model's performance across key metrics</p>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {metrics.map((metric) => (
          <Card key={metric.name}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">{metric.name}</span>
                <Badge variant="secondary">{(metric.value * 100).toFixed(1)}%</Badge>
              </div>
              <Progress value={metric.value * 100} className="h-2" />
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Performance Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 bg-gray-100 rounded-lg flex items-center justify-center">
            <span className="text-gray-500">📈 Chart visualization would go here</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function CodeSandbox() {
  const [code, setCode] = useState(`# Boston Housing Price Prediction
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load sample data
print("Loading Boston Housing dataset...")
print("Dataset loaded successfully!")`)

  const [output, setOutput] = useState("")

  const runCode = () => {
    // Simulate code execution
    setOutput("Loading Boston Housing dataset...\nDataset loaded successfully!\n\nExecution completed!")
  }

  return (
    <div className="space-y-6">
      <div>
        <h4 className="font-semibold mb-2">Live ML Code Editor</h4>
        <p className="text-sm text-gray-600">Write and execute machine learning code</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium">Code Editor</span>
            <Button onClick={runCode} size="sm">
              <Play className="h-4 w-4 mr-2" />
              Run Code
            </Button>
          </div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="w-full h-64 p-4 font-mono text-sm border rounded-lg bg-gray-900 text-green-400"
            placeholder="Write your Python code here..."
          />
        </div>

        <div>
          <span className="font-medium">Output</span>
          <div className="w-full h-64 p-4 font-mono text-sm border rounded-lg bg-gray-100 mt-2 overflow-auto">
            {output || "Click 'Run Code' to see output..."}
          </div>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Code Templates</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCode("# Data Loading Template\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())")
              }
            >
              Data Loading
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCode(
                  "# Model Training Template\nfrom sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)",
                )
              }
            >
              Model Training
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCode(
                  "# Evaluation Template\nfrom sklearn.metrics import accuracy_score\nscore = accuracy_score(y_true, y_pred)\nprint(f'Accuracy: {score}')",
                )
              }
            >
              Evaluation
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
