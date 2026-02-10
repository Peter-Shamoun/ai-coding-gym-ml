import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import ExampleTask from "@/components/ExampleTask";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
        <ExampleTask />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
