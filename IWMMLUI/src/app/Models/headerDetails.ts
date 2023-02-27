export interface IHeaderDetails{
    name:string;
    dataType:string;
}

export class HeaderDetails implements IHeaderDetails{
    name!:string;
    dataType!:string;
}