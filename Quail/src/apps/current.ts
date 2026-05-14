import { APP_NAME } from "../config.js";
import { quailApp } from "./quail/app.js";
import type { AppDefinition } from "./types.js";

const baseApp: AppDefinition = {
	id: APP_NAME,
	title: APP_NAME,
	description: "AI coding assistant with read, bash, edit, write tools",
	appendDateToCustomPrompt: true,
};

const registeredApps: readonly AppDefinition[] = [quailApp];

export const currentApp: AppDefinition = registeredApps.find((app) => app.id === APP_NAME) ?? baseApp;

export function isCurrentApp(appId: string): boolean {
	return currentApp.id === appId;
}
