## Git flow 主要三個Senario

- Check Merge vs. Rebase 
- use `git checkout` to change branch
  - Commit some changes
  - stash the change which hasn't commited. (`git stash -u`)
  - push the commit up to `remote`
  - Pull request the branch
  - Pop the stash out and continue doing development (`git stash pop`)
    
### Senario 1 (development)
當要開始一個新功能開發時，可以創建feature branch，等待這個功能開發完成，確定應用到新版本終究合併回develop。
`Rule：從develop創建，最後合併回develop`
`Branch name rule: feature/*;`
`很多地方會使用feature- or feature/`

1. create branch from develop
    `$ git checkout -b feature/test develop` (從develop 拉一分支出來)
    `$ git push origin feature/test`
    
2. 切換到develop 合併feature/test    
    `$ git checkout develop`
    `$ git merge --no-ff feature/test` (`--no-ff` 為創建一個新的commit用於合併的操作，可以避避免丟失該feature branch的歷史存在信息)
    
3. 移除本地和遠端的 feature/test
    `$ git branch -d feature/test`
    `$ git push origin --delete feature/test`
    

### Senario 2 (Release branch 預發布分支)
- 用來做新版本發布前的工作準備，可以在release branch 上做一些發布前的準備。同時，部會影響develop branch 上next version 的新功能開發
- Rule: 從 develop 分支創建，最終合併回 develop & master`
- Branch name rule: release-*
- 流程
    1. create branch 
        `$ git checkout -b release-1.1 develop`
        `$ git push origin release-1.1`
        
    2. 切換到 master 合併 release-1.1
        `$ git checkout master`
        `$ git merge --no-ff release-1.1`
        `$ git tag -a 1.1`
        `$ git push origin 1.1`
        當預示著我們可以發布，打上對應版本號，再push 到remote
        
    3. 切換到develop 合併release-1.1
        `$ git checkout develop`
        `$ git merge --no-ff release-1.1`
        
    4. 移除local & remote的release-1.1
        `$ git branch -d release-1.1`
        `$ git push origin --delete release-1.1`
        

### Scenario 3 (Hotfix branch)

Rule: 從master上當前版本號的tag切出，從最新的master上創建，最終合併回develop & master
Branch name rule: hotfix-*

1. create fixbug branch
    `$ git checkout -b fixbug-1.1.1 master`
    `$ git push origin fixbug-1.1.1`
    
2. 切換到master合併fixbug-1.1.1
    `$git checkout master`
    `$ git merge --no-ff fixbug-1.1.1`
    `$ git tag -a 1.1.1`
    `$ git push origin 1.1.1`
    Bug 修完後，合併回master並打上版本號
    
3. 切換到develop合併fixbug-1.1.1
    `$ git checkout develop`
    `$ git merge --no-ff fixbug-1.1.1`
    
4. 移除local & remote fixbug-1.1.1
    `$ git branch -d fixbug-1.1.1`
    `$ git push origin --delete fixbug-1.1.1`
    

只是**一種規範**，讓整個團隊遵守某個工作流程的規範，比較不會發生問題。
