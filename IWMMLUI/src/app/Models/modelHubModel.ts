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
        useCase: "Predictive Maintenance Of Ion Mill Etch machine",
        description: "Long short-term memory (LSTM) For Time Series Forecasting And Machine Learning for Failure Classification.",
        imagePath: "assets/modelHubImageFiles/LSTM.png",
        htmlFilePath:"../../assets/modelHubHtmlFiles/Predictive MaintenanceOfIonMillEtchmachine/lstmModel.html ",
        modelDownloadPath: "",
        modelDownloadName: "PredictiveMaintenanceOfIonMillEtchmachine_Model.zip",
        sourceCodeDownloadPath: "",
        sourceCodeDownloadName: "PredictiveMaintenanceOfIonMillEtchmachine_SourceCode.zip",
        datasetDownloadPath: "",
        datasetDownloadName: "PredictiveMaintenanceOfIonMillEtchmachine_Dataset.zip"
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
    

 
]