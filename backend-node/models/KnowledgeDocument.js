import mongoose from 'mongoose';

const knowledgeDocumentSchema = new mongoose.Schema({
    sport: {
        type: String,
        required: true,
        index: true // بنعمل إندكس عادي هنا عشان نسرع الفلترة بالرياضة
    },
    topic: {
        type: String,
        default: 'training_knowledge'
    },
    content: {
        type: String,
        required: true
    },
    embedding: {
        type: [Number], // الـ Vector هيتخزن كـ Array of Numbers
        required: true
    },
    created_at: {
        type: Date,
        default: Date.now
    }
});

export const KnowledgeDocument = mongoose.model('KnowledgeDocument', knowledgeDocumentSchema);