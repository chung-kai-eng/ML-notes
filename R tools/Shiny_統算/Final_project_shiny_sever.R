# self-defined function
standardized <- function(data, from, to){
  for(i in from:to){
    data[ ,i] <- scale(data[ ,i])
  }
  #print(head(data))
  return(data)
}
#### eda ####
GKgamma.ztest <- function(dataset, namex, namey){
  x <- dataset[ , which(colnames(dataset) == namex)]
  y <- dataset[ , which(colnames(dataset) == namey)] 
  temp = data.frame(x,y)
  names(temp) = c(namex,namey)
  tab = table(temp)
  G = GoodmanKruskalGamma(tab, conf.level = 0.95)
  cdpairs = ConDisPairs(tab)[3:4] %>% unlist()
  test = (as.numeric(G[1]) * sqrt((cdpairs[1] + cdpairs[2]) / (length(x) * (1 - as.numeric(G[1])^2)))) %>% as.numeric()
  results = ifelse(abs(test) < qnorm(0.975),"not correlated","correlated")
  return(list(list(proportion = prop.table(tab, margin = 1)),
              GoodmanKruskalGamma = G,
              test.statistics = test,
              results = results
             )
        )
  #data.frame(prop.table(x, margin = 1))
  # print("proportion: ")
  # print(tab/length(x))
  # print("GoodmanKruskalGamma")
  # print(G)
  # print("test.statistics")
  # print(test)
  # print("results")
  # print(results)
}

############################## server ##############################
server <- function(input, output, session) {
  ############################ Page 1 ##############################
  #### Input Data ####
  raw_data <- reactive({
    infile <- input$file1
    if(is.null(infile)) return(NULL)
    else{
      dataset <- read.csv(infile$datapath)
      dataset$bmi = round(dataset$weight / (dataset$height / 100) ^ 2, 2)
      dataset <- mutate(dataset, gender = factor(gender), cholesterol = factor(cholesterol),
                        gluc = factor(gluc), smoke = factor(smoke), alco = factor(alco),
                        active = factor(active), cardio = factor(cardio)) %>%
        select(id:age, height:ap_lo, bmi, gender, cholesterol:cardio)
    }
    return(dataset)
  })

  output$table1 <- DT::renderDataTable({
    DT::datatable(raw_data(), options = list(searching = F, autoWidth = TRUE, scrollX = T))
  })
  
  ############################## Page2 ##############################
  output$dataset <- renderInfoBox({
    infoBox(
      title = "Dataset", value = "cardio", icon = icon("table"), color = "aqua")
  })
  output$num_row <- renderInfoBox({
    dataset <- raw_data()
    infoBox(
      title = "Rows", value = nrow(dataset), icon = icon("poll-h"), color = "yellow")
  })
  output$num_col <- renderInfoBox({
    dataset <- raw_data()
    infoBox(
      title = "Columns", value = ncol(dataset), icon = icon("poll"), color = "green")
  })
  output$summary <- renderPrint({
    dataset <- raw_data()
    dataset <-  dataset[,-1]
    summary(dataset)
  })
  output$corrplot <- renderPlot({
    dataset <- raw_data()
    names(dataset)[names(dataset) == "cholesterol"] = "chole"
    cor_data.cor = cor(dataset[,2:7], method = c("pearson")) # default: pearson
    corrplot.mixed(cor_data.cor, upper = "square", lower = "number")
  })
  output$qt <- renderUI({
    dataset <- raw_data()
    selectInput("qt_f", "Quantitative:", choices = colnames(dataset[2:7]))
  })
  output$ql <- renderUI({
    dataset <- raw_data()
    selectInput("ql_f", "Qualitative:", choices = colnames(dataset[8:14]))
  })
  ######## quant hist #######
  output$hist_1 <- renderPlot({
    dataset <- raw_data()
    ggplot(dataset, aes_string(x = input$qt_f)) + 
      geom_histogram(col = "#5599FF", fill = "#5599FF", alpha = 0.6, bins = 10)
    
  })
  ####### quant boxplot #######
  output$boxplot_1 <- renderPlot({
    dataset <- raw_data()
    # dataset[ ," "] is a vector, dataset[" "] is a list
    dataset_i <- dataset[ , which(colnames(dataset) == input$qt_f)]
    boxplot(dataset_i, xlab = as.character(input$qt_f), col = "#FFBB66", border = "#FFAA33", las = 2)
  })
  ###### qual barplot #######
  output$bar_qual <- renderPlot({
    dataset <- raw_data()
    ggplot(data = dataset, aes_string(input$ql_f)) + 
      geom_bar(stat = "count", fill = "#9999FF", alpha = 0.4)
  })
  
  output$corr_qual <- renderPrint({
    data <- raw_data()
    g <- GKgamma.ztest(data, input$ql_f, "cardio")
    print(g)
  })

  output$bar_cardio <- renderPlot({
    dataset <- raw_data()
    qual_var <- dataset[ , which(colnames(dataset) == input$ql_f)]
    cardiovascular <- dataset$cardio
    x = table(cardiovascular, qual_var)
    prop_tab = data.frame(prop.table(x, margin = 1))
    cat("Proportion between cardiovascular & ", input$ql_f, "\n")
    print(prop_tab)
    names(prop_tab) <- c("cardio", input$ql_f, "prop")
    ggplot(prop_tab, aes_string(x = "cardio", y = "prop", fill = input$ql_f)) +
      geom_bar(stat = "identity", alpha = 0.6)
    
  })
  
  output$boxplot_2 <- renderPlot({
    dataset <- raw_data()
    dataset_x <- dataset[ , which(colnames(dataset) == input$ql_f)]
    dataset_y <- dataset[ , which(colnames(dataset) == input$qt_f)]
    ggplot(dataset, aes(x = as.vector(dataset_x), y = as.vector(dataset_y))) + 
      geom_boxplot(col ="#00AA00", fill = "#00AA00", alpha = 0.4) +
      ggtitle(paste(as.character(input$qt_f), "vs.", as.character(input$ql_f))) + 
      ylab(paste(as.character(input$qt_f))) +
      xlab(as.character(input$ql_f))
  })
  output$anova <- renderPrint({
    dataset <- raw_data()
    qual_var <- unlist(dataset[ , which(colnames(dataset) == input$ql_f)])
    quan_var <- unlist(dataset[ , which(colnames(dataset) == input$qt_f)])
    name_anova <- paste0("One way ANOVA for ", input$qt_f, " ~ ", input$ql_f)
    print(name_anova)
    aov_1 = aov(quan_var ~ qual_var)
    summary(aov_1)
  })

  ############################ Page 3 ##############################
  ###########################################
  # Clustering
  ###########################################
  # output$classifier <- renderText({
  #   
  # 
  # })
  output$qt <- renderUI({
    dataset <- raw_data()
    selectInput("qt_f", "Quantitative:", choices = colnames(dataset[2:7]))
  })
  output$kmean <- renderPrint({
    dataset <- raw_data()
    data_clus <- dataset %>% select(id:age, height:ap_lo, gender, cholesterol:cardio)
    std_data <- standardized(data_clus, 2, 6)
    # lambda: parameter > 0 to trade off between Euclidean distance of numeric variables
    # and simple matching coefficient between categorical variables.
    sug_lambda <- lambdaest(dataset[2:12], num.method = 2, fac.method = 2, outtype = "numeric")
    # determine which factors are important
    clus_1 <- kproto(dataset[, 2:12], 2, lambda = sug_lambda, iter.max = 100)
    summary(clus_1)
    # visualization of a k-prototypes clustering result for cluster interpretation
    # for numerical variables boxplot and for categorical variables barplots
    clprofiles(clus_1, dataset)
  })
    
    
  
 
  
} # server end
