# LassieRAG

## Environment

1. Install poetry

    - Windows

      ```powershell
      (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
      ```

    - Linux/ MacOs

      ```bash
      curl -sSL https://install.python-poetry.org | python3 -
      ```

2. Create virtual environment

   use absolute path for python if needed

    ```bash
    poetry env use python3.12
    poetry shell
    ```

3. Install dependencies

    ```bash
    poetry install --with dev
    ```