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
    }
 
]