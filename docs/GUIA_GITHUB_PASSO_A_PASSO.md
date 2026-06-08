# Guia GitHub passo a passo

## Primeiro commit

1. Extraia `versao-0.1-dashboard-streamlit-inicial.zip` em uma pasta vazia.
2. Abra o terminal nessa pasta.
3. Execute:

```powershell
git init -b main
git add .
git commit -m "chore: cria estrutura inicial do dashboard streamlit"
git remote add origin URL_DO_REPOSITORIO
git push -u origin main
```

## Commits seguintes

Para cada ZIP seguinte:

1. extraia o conteúdo sobre a mesma pasta;
2. aceite substituir os arquivos existentes;
3. confira as alterações com `git status` e `git diff --stat`;
4. registre o commit correspondente conforme `docs/PLANO_DE_COMMITS.md`;
5. execute `git push`.

Não existe arquivo `VERSION.md` dentro das etapas. A ordem é indicada exclusivamente pelos nomes dos ZIPs.
