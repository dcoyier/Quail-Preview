import { APP_NAME } from "../config.js";
import { quailApp } from "./quail/app.js";
const baseApp = {
    id: APP_NAME,
    title: APP_NAME,
    description: "AI coding assistant with read, bash, edit, write tools",
    appendDateToCustomPrompt: true,
};
const registeredApps = [quailApp];
export const currentApp = registeredApps.find((app) => app.id === APP_NAME) ?? baseApp;
export function isCurrentApp(appId) {
    return currentApp.id === appId;
}
//# sourceMappingURL=current.js.map