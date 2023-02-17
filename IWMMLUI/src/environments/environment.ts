// This file can be replaced during build by using the `fileReplacements` array.
// `ng build` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

export const environment = {
  production: false,
  clientId: 'b1924ee5-eb10-4a30-bd70-a9837fcf73c7',
  authority: 'https://login.microsoftonline.com/687f51c3-0c5d-4905-84f8-97c683a5b9d1/',
  redirectUrl: 'http://localhost:4200/',
  apiUrl:'http://127.0.0.1:5000/',
  graphUrl: 'https://graph.microsoft.com/v1.0'
};

/* 
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/plugins/zone-error';  // Included with Angular CLI.

