"""指定したnotebookを別カーネルで実行し、進捗と実行結果を保存する。"""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys

import nbformat
from nbclient import NotebookClient


class LiveClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg.get("msg_type")=="stream":
            print(msg["content"]["text"],end="",flush=True)
        return super().process_message(msg,cell,cell_index)


root=Path(__file__).resolve().parents[1]
status_path=root/"results"/os.environ.get("PAPER1_STATUS_FILE","execution_status.json")
state={"pid":os.getpid(),"status":"running","notebooks":sys.argv[1:],"completed":[],
       "started_at":datetime.now(timezone.utc).isoformat(),"current":None}
def save_state():
    state["updated_at"]=datetime.now(timezone.utc).isoformat()
    temporary=status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,indent=2)+"\n")
    temporary.replace(status_path)
save_state()
for filename in sys.argv[1:]:
    path=root/"notebooks"/filename
    notebook=nbformat.read(path,as_version=4)
    print(f"Executing {path.name}",flush=True)
    state["current"]=filename
    save_state()
    try:
        timeout=None if filename=="06_end_to_end.ipynb" else 14400
        LiveClient(notebook,timeout=timeout,kernel_name="paper1",resources={"metadata":{"path":str(root)}}).execute()
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        save_state()
        raise
    finally:
        nbformat.write(notebook,path)
    print(f"Finished {path.name}",flush=True)
    state["completed"].append(filename)
    save_state()
state.update(status="completed",current=None)
save_state()
