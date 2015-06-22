args=(commandArgs(TRUE))


if(length(args)==0){
    print("No arguments supplied.")
    fn="data.csv"
}else{
    fn=args[1]
}

data = read.csv(fn, header=TRUE)
fn_noext = sub("\\.[[:alnum:]]+$", "", basename(as.character(fn)))
image_fn = paste(fn_noext, '.jpg', sep="")
jpeg(image_fn)
boxplot(data)
dev.off()
