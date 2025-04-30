import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { motion } from "framer-motion";

export function TestimonialSection() {
  const testimonials = [
    {
      quote: "ReSync transformed how we collaborate across disciplines. The AI recommendations are a game-changer!",
      author: "Dr. Priya Sharma",
      role: "Research Lead at VESIT",
      avatar: "https://randomuser.me/api/portraits/women/1.jpg",
      company: "vesit.edu",
    },
    {
      quote: "A centralized platform that saved us hours of searching for datasets. Highly recommend for interdisciplinary work!",
      author: "Amirta Patel",
      role: "Data Scientist at IIT Bombay",
      avatar: "https://randomuser.me/api/portraits/men/2.jpg",
      company: "iitb.ac.in",
    },
    {
      quote: "The real-time collaboration tools and secure environment made our hackathon project a success!",
      author: "Sneha Kapoor",
      role: "Student at VESIT",
      avatar: "https://randomuser.me/api/portraits/women/3.jpg",
      company: "vesit.edu",
    },
  ];

  return (
    <section className="relative py-20 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-white to-gray-50" />
      <div className="absolute inset-0 bg-grid-gray-100/[0.2] bg-[size:20px_20px]" />
      
      <div className="container relative px-4 mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.span 
            className="inline-block px-4 py-2 mb-4 text-sm font-semibold tracking-wider text-blue-600 uppercase bg-blue-100 rounded-full"
            whileHover={{ scale: 1.05 }}
          >
            Testimonials
          </motion.span>
          <h2 className="mb-4 text-4xl font-bold text-gray-800 md:text-5xl">
            Trusted by Researchers
          </h2>
          <p className="mx-auto text-xl text-gray-600 md:max-w-xl">
            Hear from researchers who have transformed their work with Research<span className="font-hindi">साथी</span>.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.2 }}
              whileHover={{ y: -5 }}
              className="relative p-8 bg-white rounded-2xl shadow-xl"
            >
              <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 opacity-5 rounded-full blur-2xl" />
              
              <svg className="w-10 h-10 mb-4 text-blue-500 opacity-20" fill="currentColor" viewBox="0 0 32 32">
                <path d="M16 0C7.16 0 0 7.16 0 16s7.16 16 16 16 16-7.16 16-16S24.84 0 16 0zm6.4 14.4h-3.2v3.2h3.2v3.2h-6.4v-9.6h6.4v3.2zm-9.6 0h-3.2v3.2h3.2v3.2H6.4v-9.6h6.4v3.2z"/>
              </svg>
              
              <p className="mb-6 text-lg italic text-gray-700">
                "{testimonial.quote}"
              </p>

              <div className="flex items-center gap-4">
                <Avatar className="w-12 h-12 border-2 border-white shadow-lg">
                  <AvatarImage src={testimonial.avatar} />
                  <AvatarFallback className="bg-gradient-to-br from-blue-600 to-purple-600">
                    {testimonial.author.split(" ").map((n) => n[0]).join("")}
                  </AvatarFallback>
                </Avatar>
                
                <div>
                  <h4 className="font-semibold text-gray-900">{testimonial.author}</h4>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <span>{testimonial.role}</span>
                    <span>•</span>
                    <a href={`https://${testimonial.company}`} className="hover:text-blue-600">
                      {testimonial.company}
                    </a>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}