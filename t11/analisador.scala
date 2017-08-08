import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object Analisador {

  // Args = path/to/text0.txt path/to/text1.txt
  def main(args: Array[String]) {
    val inputFile0 = args(0)
    val inputFile1 = args(1)

    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Contagem de Palavra"))

    println("TEXT1")

    // read first text file and split into lines
    val input0 =  sc.textFile(inputFile0)
    // Split up into words.
    val words0 = input0.flatMap(line => line.toLowerCase.replaceAll("[,.!?:;]'","").split(" ")).filter(word => {word.length() > 3})
    // Transform into word and count.
    val counts0 = words0.map(word => (word, 1)).reduceByKey{case (x, y) => x + y}

    // Order words by counts
    val counts_swap0 = counts0.map(_.swap)
    val count_order0 = counts_swap0.sortByKey(false,1)
    val count_order_swap0 = count_order0.map(_.swap)
    // Get all words that have more of 100 counts
    val more_100_0 = count_order_swap0.filter(word => {word._2 > 100}).map(word => word._1)
    // Get top 5 of words with more counts
    val top_5_0 = count_order_swap0.take(5)
    // Print top five
    top_5_0.foreach(top => println(top._1 + "=" + top._2))

    println("TEXT2")
    // read first text file and split into lines
    val input1 =  sc.textFile(inputFile1)
    // Split up into words.
    val words1 = input1.flatMap(line => line.toLowerCase.replaceAll("[,.!?:;]","").split(" ")).filter(word => {word.length() > 3})
    // Transform into word and count.
    val counts1 = words1.map(word => (word, 1)).reduceByKey{case (x, y) => x + y}

    // Order words by counts
    val counts_swap1 = counts1.map(_.swap)
    val count_order1 = counts_swap1.sortByKey(false,2)
    val count_order_swap1 = count_order1.map(_.swap)
    // Get all words that have more of 100 counts
    val more_100_1 = count_order_swap1.filter(word => {word._2 > 100}).map(word => word._1)
    // Get top 5 of words with more counts
    val top_5_1 = count_order_swap1.take(5)

    // Print top five
    top_5_1.foreach(top => println(top._1 + "=" + top._2))


    println("COMMON")
    // Get intersection
    val intersection = more_100_0.intersection(more_100_1).collect().sorted
    // Print intersection words
    intersection.foreach(top => println(top))


  }

}