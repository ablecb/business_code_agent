const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  getConfig: () => ipcRenderer.invoke("config:get"),
  setConfig: (partial) => ipcRenderer.invoke("config:set", partial),
  openProjectDirDialog: () => ipcRenderer.invoke("dialog:openProjectDir"),
  getProjectLocalRoots: () => ipcRenderer.invoke("projectRoots:get"),
  getProjectLocalRoot: (projectId) => ipcRenderer.invoke("projectRoots:getOne", projectId),
  setProjectLocalRoot: (projectId, rootPath) =>
    ipcRenderer.invoke("projectRoots:set", projectId, rootPath),
  useProjectRoot: (projectId) => ipcRenderer.invoke("executor:useProject", projectId),
  getLocalRoot: () => ipcRenderer.invoke("executor:getRoot"),
  setLocalRoot: (rootPath) => ipcRenderer.invoke("executor:setRoot", rootPath),
  runAction: (action) => ipcRenderer.invoke("executor:run", action),
  showItemInFolder: (targetPath) =>
    ipcRenderer.invoke("shell:showItemInFolder", targetPath),
  onExecutorLog: (handler) => {
    const fn = (_event, payload) => handler(payload);
    ipcRenderer.on("executor:log", fn);
    return () => ipcRenderer.removeListener("executor:log", fn);
  },
  platform: process.platform,
});
