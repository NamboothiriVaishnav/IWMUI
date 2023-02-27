export interface IDataModel {
    id: number,
    useCase: string,
    htmlFilePath: string
}

export const DataModel: IDataModel[] = [
    {
        id: 1,
        useCase: "Failure prediction Of Ion Mill Etch Machine",
        htmlFilePath: "../../assets/dataModelHtmlFiles/FailurePredectionOfIonMillEtchMachine-dataModel.html",
    }
]