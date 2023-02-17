import { IGenericType } from './genericType';
export interface IColumnDataType extends IGenericType {
}

export const ColumnDataTypes: IColumnDataType[] = [
  {
    name: 'string',
    value: 'string'
  },
  {
    name: 'integer',
    value: 'integer'
  },
  {
    name: 'datetime',
    value: 'datetime'
  },
  {
    name: 'boolean',
    value: 'boolean'
  },
  {
    name: 'float',
    value: 'float'
  }
];
