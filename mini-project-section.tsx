"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useState } from "react"
import { CheckCircle, Cpu, DatabaseZap, FolderGit2, Layers3, PlayCircle } from "lucide-react"
import type { JSX } from "react/jsx-runtime"

const steps = [
  {
    id: "data_discovery",
    title: "Data Discovery",
    description: "Understand the dataset and perform EDA.",
    icon: DatabaseZap,
  },
  {
    id: "data_versioning",
    title: "Data Versioning with DVC",
    description: "Track dataset and changes using DVC.",
    icon: FolderGit2,
  },
  {
    id: "reproducibility",
    title: "Reproducibility",
    description: "Ensure reproducibility through pipeline structure.",
    icon: Layers3,
  },
  {
    id: "learning_type",
    title: "Supervised vs Unsupervised",
    description: "Compare learning types and use cases.",
    icon: Cpu,
  },
  {
    id: "scikit_dvc",
    title: "Hands-on with Scikit-learn & DVC",
    description: "Build a real project using both tools.",
    icon: PlayCircle,
  },
]

const stepContent: Record<string, JSX.Element> = {
  data_discovery: (
    <CardContent>
      <p className="mb-2">
        We're using the <strong>Boston Housing dataset</strong> to predict house prices.
      </p>
      <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">
        {`from sklearn.datasets import load_boston  # deprecated; consider fetch_california_housing instead
import pandas as pd

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["PRICE"] = data.target
print(df.head())`}
      </pre>
    </CardContent>
  ),
  data_versioning: (
    <Tabs defaultValue="setup" className="w-full">
      <TabsList>
        <TabsTrigger value="setup">DVC Setup</TabsTrigger>
        <TabsTrigger value="track">Track Dataset</TabsTrigger>
        <TabsTrigger value="commit">Commit Changes</TabsTrigger>
      </TabsList>
      <TabsContent value="setup">
        <CardContent>
          <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">
            {`pip install dvc
dvc init`}
          </pre>
        </CardContent>
      </TabsContent>
      <TabsContent value="track">
        <CardContent>
          <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">{`dvc add data/boston.csv`}</pre>
        </CardContent>
      </TabsContent>
      <TabsContent value="commit">
        <CardContent>
          <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">
            {`git add data/boston.csv.dvc .gitignore
git commit -m "Track data with DVC"`}
          </pre>
        </CardContent>
      </TabsContent>
    </Tabs>
  ),
  reproducibility: (
    <CardContent>
      <p className="mb-2">DVC helps in making ML pipelines reproducible. Here's how to define a stage:</p>
      <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">
        {`dvc run -n train_model \\
  -d train.py -d data/boston.csv \\
  -o model.pkl \\
  -p alpha \\
  -m metrics.json \\
  python train.py`}
      </pre>
    </CardContent>
  ),
  learning_type: (
    <CardContent>
      <p>
        <strong>Supervised Learning:</strong> Uses labeled data (e.g., price prediction)
      </p>
      <p>
        <strong>Unsupervised Learning:</strong> No labels, finds hidden patterns (e.g., clustering)
      </p>
    </CardContent>
  ),
  scikit_dvc: (
    <CardContent>
      <p className="mb-2">Here's a mini project using Scikit-learn:</p>
      <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto">
        {`from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

X = df.drop("PRICE", axis=1)
y = df["PRICE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("MSE:", mean_squared_error(y_test, y_pred))
joblib.dump(model, "model.pkl")`}
      </pre>
    </CardContent>
  ),
}

export function MiniProjectSection() {
  const [activeStep, setActiveStep] = useState("data_discovery")
  const [completedSteps, setCompletedSteps] = useState<string[]>([])

  const markCompleted = (stepId: string) => {
    if (!completedSteps.includes(stepId)) {
      setCompletedSteps([...completedSteps, stepId])
    }
  }

  return (
    <div className="flex flex-col md:flex-row gap-4">
      <div className="md:w-1/3 space-y-2">
        {steps.map((step) => {
          const Icon = step.icon
          const isActive = activeStep === step.id
          const isCompleted = completedSteps.includes(step.id)
          return (
            <button
              key={step.id}
              onClick={() => setActiveStep(step.id)}
              className={`w-full flex justify-start text-left h-auto p-3 rounded-lg border ${
                isActive ? "border-blue-600 bg-blue-50" : "border-gray-200"
              }`}
            >
              <Icon className="w-5 h-5 mr-2" />
              <div className="flex flex-col items-start">
                <span className="text-sm font-medium">{step.title}</span>
                <span className="text-xs text-gray-500">{step.description}</span>
                {isCompleted && (
                  <span className="text-green-600 text-xs flex items-center gap-1 mt-1">
                    <CheckCircle className="w-3 h-3" /> Completed
                  </span>
                )}
              </div>
            </button>
          )
        })}
      </div>

      <div className="md:w-2/3">
        <Card>
          <CardHeader>
            <CardTitle>{steps.find((s) => s.id === activeStep)?.title}</CardTitle>
            <CardDescription>{steps.find((s) => s.id === activeStep)?.description}</CardDescription>
          </CardHeader>
          {stepContent[activeStep]}
          <CardContent className="mt-4">
            <button
              onClick={() => markCompleted(activeStep)}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm"
            >
              Mark Step as Completed
            </button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
