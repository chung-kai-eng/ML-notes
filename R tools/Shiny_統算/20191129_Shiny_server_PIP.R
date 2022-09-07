####################### Enlarge Capacity ###########################
options(shiny.maxRequestSize=1000*1024^2)
######################### sliderInput2 #############################
sliderInput2 <- function(inputId, label, min, max, value, step, from_min, from_max){
  x <- sliderInput(inputId, label, min, max, value, step)
  x$children[[2]]$attribs <- c(x$children[[2]]$attribs, 
                               "data-from-min" = from_min, 
                               "data-from-max" = from_max, 
                               "data-from-shadow" = TRUE)
  x
}
############################## server ##############################
server <- function(input, output) {
  ############################ Page 1 ##############################
  #### Input Data ####
  para_data <- reactive({
    infile4 <- input$file4
    if(is.null(infile4)) return(NULL)
    read.csv(infile4$datapath)
  })
  ### Page1 Data(try)
  all_clean_data <- reactive({
    infile5 <- input$file5
    if(is.null(infile5)) return(NULL)
    read.csv(infile5$datapath)
  })
  ### all_clean_data()->by_lot_data()
  output$table1 <- DT::renderDataTable({
    DT::datatable(all_clean_data(), options = list(searching = FALSE, autoWidth = TRUE, scrollX = T))
  })
  
  ############################## Page2 ##############################
  ### Product condition ###
  Layer1 <- reactive({
    clean_data <- all_clean_data()
    if(is.null(clean_data)){
      return()
    }else{
      clean_data[,"Size"] <- paste(clean_data$PKG_X, clean_data$PKG_Y, sep = "-")
      L1_data<-clean_data[which(clean_data$WIRE_DIAMETER == input$W_D),]
    }
    return(L1_data)
  })
  Layer2 <- reactive({
    L1_data <- Layer1()
    if(is.null(L1_data)){
      return()
    }else{
      row.names(L1_data)<-c(1:nrow(L1_data))
      L1_data[,"temp_ratio"]<-apply(L1_data[,c(52,53)],1,max)/apply(L1_data[,c(52,53)],1,min)
      BAD_FRAME<-subset(L1_data, (shape == "SQUARE" & PKG_X >= 6|temp_ratio>1.5)&(Vendor=="1140"|Vendor=="463"|Vendor=="5415"|Vendor=="STW"))
      GOOD_FRAME <- L1_data[-as.numeric(row.names(BAD_FRAME)),]
      if((input$Vendor == "1140"|input$Vendor == "463"|input$Vendor == "5415"|input$Vendor == "STW")&
         (input$Size == "5-9"|input$Size == "6-6"|input$Size == "7-7"|input$Size == "8-8"|input$Size == "9-9"|input$Size == "10-10")){
        return(BAD_FRAME)
      }else{
        return(GOOD_FRAME)
      }
    }
  })
  Layer3 <- reactive({
    L2_data <- Layer2()
    if(is.null(L2_data)){
      return()
    }else{
      L3_data <- L2_data[which(L2_data$CAPILLARY_TYPE == input$Cap),]
    }
    return(L3_data)
  })
  ### Show Selectize List ###
  output$W_D <- renderUI({
    clean_data <- all_clean_data()
    selectInput("W_D", label = NULL,choices = sort(clean_data$WIRE_DIAMETER))
  })
  output$Vendor <- renderUI({
    L1_data <- Layer1()
    selectInput("Vendor", "GROUP1:", choices = as.character(L1_data$Vendor))
  })
  output$Size <- renderUI({
    L1_data <- Layer1()
    selectInput("Size", "GROUP2:", choices = sort(L1_data$Size))
  })
  output$Cap <- renderUI({
    L2_data <- Layer2()
    selectInput("Cap", label = NULL, choices = as.character(L2_data$CAPILLARY_TYPE))
  })
  
  #################################### Parameter Tuning ###################################
  ##### sliderbar  input
  # para_data() must be change to raw_data() -> only for project demo
  slider_data <- eventReactive(input$start,{
    raw_data<-para_data()
    return(raw_data)
  })
  
  observeEvent(input$start,{
    sdata <- slider_data()
    i <- 1
    # while(i <= nrow(sdata)){
    output$para_tune <- renderUI({
      fluidRow(
        column(4,sliderInput2(inputId = as.character(sdata[i,1]),label = as.character(sdata[i,8]),
                              min = sdata[i,3], max = sdata[i,4], value = sdata[i,2],
                              from_min = sdata[i,5], from_max = sdata[i,6], step=sdata[i,7])), # column 1 end
        column(4,sliderInput2(inputId = as.character(sdata[i+1,1]),label = as.character(sdata[i+1,8]),
                              min = sdata[i+1,3], max = sdata[i+1,4], value = sdata[i+1,2],
                              from_min = sdata[i+1,5], from_max = sdata[i+1,6], step=sdata[i+1,7])), # column 2 end
        column(4,sliderInput2(inputId = as.character(sdata[i+2,1]),label = as.character(sdata[i+2,8]),
                              min = sdata[i+2,3], max = sdata[i+2,4], value = sdata[i+2,2],
                              from_min = sdata[i+2,5], from_max = sdata[i+2,6], step=sdata[i+2,7])), # column 3 end
        column(4,sliderInput2(inputId = as.character(sdata[i+3,1]),label = as.character(sdata[i+3,8]),
                              min = sdata[i+3,3], max = sdata[i+3,4], value = sdata[i+3,2],
                              from_min = sdata[i+3,5], from_max = sdata[i+3,6], step=sdata[i+3,7])), # column 4 end
        column(4,sliderInput2(inputId = as.character(sdata[i+4,1]),label = as.character(sdata[i+4,8]),
                              min = sdata[i+4,3], max = sdata[i+4,4], value = sdata[i+4,2],
                              from_min = sdata[i+4,5], from_max = sdata[i+4,6], step=sdata[i+4,7])), # column 5 end
        column(4,sliderInput2(inputId = as.character(sdata[i+5,1]),label = as.character(sdata[i+5,8]),
                              min = sdata[i+5,3], max = sdata[i+5,4], value = sdata[i+5,2],
                              from_min = sdata[i+5,5], from_max = sdata[i+5,6], step=sdata[i+5,7])), # column 6 end
        column(4,sliderInput2(inputId = as.character(sdata[i+6,1]),label = as.character(sdata[i+6,8]),
                              min = sdata[i+6,3], max = sdata[i+6,4], value = sdata[i+6,2],
                              from_min = sdata[i+6,5], from_max = sdata[i+6,6], step=sdata[i+6,7])), # column 7 end
        column(4,sliderInput2(inputId = as.character(sdata[i+7,1]),label = as.character(sdata[i+7,8]),
                              min = sdata[i+7,3], max = sdata[i+7,4], value = sdata[i+7,2],
                              from_min = sdata[i+7,5], from_max = sdata[i+7,6], step=sdata[i+7,7])), # column 8 end
        column(4,sliderInput2(inputId = as.character(sdata[i+8,1]),label = as.character(sdata[i+8,8]),
                              min = sdata[i+8,3], max = sdata[i+8,4], value = sdata[i+8,2],
                              from_min = sdata[i+8,5], from_max = sdata[i+8,6], step=sdata[i+8,7])), # column 9 end
        column(4,sliderInput2(inputId = as.character(sdata[i+9,1]),label = as.character(sdata[i+9,8]),
                              min = sdata[i+9,3], max = sdata[i+9,4], value = sdata[i+9,2],
                              from_min = sdata[i+9,5], from_max = sdata[i+9,6], step=sdata[i+9,7])), # column 10 end
        column(4,sliderInput2(inputId = as.character(sdata[i+10,1]),label = as.character(sdata[i+10,8]),
                              min = sdata[i+10,3], max = sdata[i+10,4], value = sdata[i+10,2],
                              from_min = sdata[i+10,5], from_max = sdata[i+10,6], step=sdata[i+10,7])), # column 11 end
        column(4,sliderInput2(inputId = as.character(sdata[i+11,1]),label = as.character(sdata[i+11,8]),
                              min = sdata[i+11,3], max = sdata[i+11,4], value = sdata[i+11,2],
                              from_min = sdata[i+11,5], from_max = sdata[i+11,6], step=sdata[i+11,7])) # column 12 end
      ) # fluid row end
    }) # renderUI end
    #   i <- i + 2
    # } # while end
  }) # observeEvent end
  
  ##### slider input value
  sliderValues <- reactive({
    data.frame(
      Parameter = c("Parameter1",
                    "Parameter2",
                    "Parameter3",
                    "Parameter4",
                    "Parameter5",
                    "Parameter6",
                    "Parameter7",
                    "Parameter8",
                    "Parameter9",
                    "Parameter10",
                    "Parameter11",
                    "Parameter12"),
      Value = as.character(c(input$Bond2Seg_ForceEqual,
                             input$Bond2Seg_ScrubWidth,
                             input$Bond2Seg_USGEqual,
                             input$Bond2_BondGeometry,
                             input$Bond2_Energy_Offset,
                             input$Bond2_Force_Offset,
                             input$ScrubFreq_Seg1,
                             input$ScrubFreq_Seg2,
                             input$Tip_2,
                             input$C.V_2,
                             input$ScrubCycles_Seg1,
                             input$ScrubCycles_Seg2)),
      stringsAsFactors = FALSE)
  })
  
  # Show the values in an HTML table -----
  output$values <- renderTable({
    sliderValues()
  }) #value end
  
  output$txtout <- renderText({
    check_result <- sliderValues()
    if(is.null(check_result)) {
      return(NULL)
    }
    y_Tip<-input$Tip_2
    y_C.V<-input$C.V_2
    y_Seg2.Scrub.Cycles<-input$ScrubCycles_Seg2
    y_Seg1.Scrub.Cycles<-input$ScrubCycles_Seg1
    y_Seg2.Scrub.Frequency<-input$ScrubFreq_Seg2
    y_Seg1.Scrub.Frequency<-input$ScrubFreq_Seg1
    y_BondTime <- (( y_Tip / y_C.V )+
                     ((y_Seg2.Scrub.Cycles +2)*1000/y_Seg1.Scrub.Frequency)+
                     ((y_Seg2.Scrub.Cycles+2)*1000/y_Seg2.Scrub.Frequency))
  })
  
  output$barplot <- renderPlot({
    check_result <- sliderValues()
    if(is.null(check_result)) {
      return(NULL)
    }
    y_Tip<-input$Tip_2
    y_C.V<-input$C.V_2
    y_Seg2.Scrub.Cycles<-input$ScrubCycles_Seg2
    y_Seg1.Scrub.Cycles<-input$ScrubCycles_Seg1
    y_Seg2.Scrub.Frequency<-input$ScrubFreq_Seg2
    y_Seg1.Scrub.Frequency<-input$ScrubFreq_Seg1
    y_BondTime <- (( y_Tip / y_C.V )+
                     ((y_Seg2.Scrub.Cycles +2)*1000/y_Seg1.Scrub.Frequency)+
                     ((y_Seg2.Scrub.Cycles+2)*1000/y_Seg2.Scrub.Frequency))
    barplot(y_BondTime, main="2nd Bond Time", horiz=TRUE, xlim = c(0,60),ylim = c(0,4),
            names.arg=c("(ms)"), cex.names=0.8,cex.axis = 0.5)
  })
  
  ####### repair model result ing ########
  output$alarmrate_SHTL<-renderText({
    if(input$Bond2Seg_ForceEqual<=333)
      x="0: safe"
    else
      x="1: alarm"
  })
  
  output$alarmrate_NSOL<-renderText({
    if(input$Bond2Seg_ScrubWidth<=100)
      x="0: safe"
    else
      x="1: alarm"
  })
  
  output$alarmrate_EFO<-renderText({
    if(input$Bond2Seg_USGEqual<=336)
      x="0: safe"
    else
      x="1: alarm"
  })
  
  ####### Reset button #########
  observeEvent(input$reset, {
    reset("form")
  })
  
} # server end
