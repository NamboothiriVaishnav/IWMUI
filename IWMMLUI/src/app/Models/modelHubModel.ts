export interface IModelHubModel {
    id: number,
    useCase: string,
    description: string,
    imagePath: string,
    htmlFilePath: string;
    modelDownloadPath: string;
    modelDownloadName: string;
    sourceCodeDownloadPath: string;
    sourceCodeDownloadName: string;
    datasetDownloadPath: string;
    datasetDownloadName: string;
}

export const ModelHubModel: IModelHubModel[] = [
    { 
        id: 1,
        useCase: "Equipment Failure Prediction Of Ion Mill Etch Machine",
        description: "Long short-term memory (LSTM) For Time Series Forecasting And Machine Learning for Failure Classification.",
        imagePath: "assets/modelHubImageFiles/LSTM.png",
        htmlFilePath:"../../assets/modelHubHtmlFiles/EquipmentFailurePredictionOfIonMillEtchMachine/lstmModel.html ",
        modelDownloadPath: "",
        modelDownloadName: "EquipmentFailurePredictionOfIonMillEtchMachine_Model.zip",
        sourceCodeDownloadPath: "",
        sourceCodeDownloadName: "EquipmentFailurePredictionOfIonMillEtchMachine_SourceCode.zip",
        datasetDownloadPath: "",
        datasetDownloadName: "EquipmentFailurePredictionOfIonMillEtchMachine_Dataset.zip"
    },
    { 
        id: 2,
        useCase: "PCB Manufacturing Defect Detection ",
        description: "Yolo-v5 for Manufactured PCB Board as a part of product quality test.",
        imagePath: "assets/modelHubImageFiles/PcbDetection.png",
        htmlFilePath:"../../assets/modelHubHtmlFiles/PcbDefectDetection/pcbModel.html",
        modelDownloadPath: "",
        modelDownloadName: "PCB Manufacturing Defect Detection_Model.zip",
        sourceCodeDownloadPath: "",
        sourceCodeDownloadName: "PCB Manufacturing Defect Detection_SourceCode.zip",
        datasetDownloadPath: "",
        datasetDownloadName: "PCB Manufacturing Defect Detection_Dataset.zip"
    },

    { 
        id: 3,
        useCase: "Processor Defect Classification ",
        description: "EfficientNet is a family of convolutional neural network architectures that were designed to achieve state-of-the-art performance while being computationally efficient.The EfficientNet architecture is based on a compound scaling method that uniformly scales the network width, depth, and resolution with a set of scaling coefficients.",
        imagePath: "assets/modelHubImageFiles/processorDefect.png",
        htmlFilePath:"../../assets/modelHubHtmlFiles/ProcessorDefectClassification/processorModel.html",
        modelDownloadPath: "",
        modelDownloadName: "PCB Manufacturing Defect Detection_Model.zip",
        sourceCodeDownloadPath: "",
        sourceCodeDownloadName: "PCB Manufacturing Defect Detection_SourceCode.zip",
        datasetDownloadPath: "",
        datasetDownloadName: "PCB Manufacturing Defect Detection_Dataset.zip"
    },
    

 
]