"""Tool declarations for Gemini Interactions API."""
from typing import Dict, Any

# Create shell tool declaration
create_shell_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "create_shell",
    "description": "Create a new shell environment for executing commands. Returns a unique shell_id that can be used to run commands in that shell.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Optional name for the shell (for debugging purposes)"
            }
        },
        "required": []
    }
}

# Run in shell tool declaration
run_in_shell_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "run_in_shell",
    "description": "Execute a command in an existing shell environment. Returns command output, exit code, and any errors.",
    "parameters": {
        "type": "object",
        "properties": {
            "shell_id": {
                "type": "string",
                "description": "The ID of the shell to run the command in"
            },
            "cmd": {
                "type": "string",
                "description": "The command to execute in the shell"
            }
        },
        "required": ["shell_id", "cmd"]
    }
}

# Read file tool declaration
read_file_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "read_file",
    "description": "Read the contents of a file from the filesystem. Returns the file contents as a string.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The absolute or relative path to the file to read"
            }
        },
        "required": ["path"]
    }
}

# Write file tool declaration
write_file_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "write_file",
    "description": "Write content to a file, overwriting if it exists. Creates the file if it doesn't exist.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to write"
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file"
            }
        },
        "required": ["path", "content"]
    }
}

# Create file tool declaration
create_file_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "create_file",
    "description": "Create a new file at the specified path. Fails if file already exists.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path where the new file should be created"
            },
            "content": {
                "type": "string",
                "description": "Initial content for the new file"
            }
        },
        "required": ["path"]
    }
}

# Delete file tool declaration
delete_file_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "delete_file",
    "description": "Delete a file from the filesystem. Fails if the file doesn't exist.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to delete"
            }
        },
        "required": ["path"]
    }
}

# Undo tool declaration
undo_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "undo",
    "description": "Undo the last file modification operation. Reverts the most recent write/create operation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# User question tool declaration
user_question_declaration: Dict[str, Any] = {
    "type": "function",
    "name": "user_question",
    "description": "Ask the user a question and wait for their response. Use this when the agent needs clarification or user input to proceed.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The question to ask the user"
            }
        },
        "required": ["text"]
    }
}